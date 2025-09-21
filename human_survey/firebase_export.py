#!/usr/bin/env python3
import argparse
import base64
import json
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

from google.cloud import firestore
from google.cloud.firestore_v1 import DocumentReference, GeoPoint
from google.oauth2 import service_account


def serialize_value(value: Any) -> Any:
    """Make Firestore types JSON-serializable."""
    if isinstance(value, datetime):
        # Firestore timestamps arrive as Python datetimes
        return value.isoformat()
    if isinstance(value, bytes):
        return {"_type": "bytes", "base64": base64.b64encode(value).decode("utf-8")}
    if isinstance(value, GeoPoint):
        return {"_type": "geopoint", "lat": value.latitude, "lng": value.longitude}
    if isinstance(value, DocumentReference):
        return {"_type": "doc_ref", "path": value.path}
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(v) for v in value]
    # bool, int, float, str, None fall through
    return value


def serialize_doc(doc_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {k: serialize_value(v) for k, v in doc_dict.items()}


def paginate_collection(coll_ref, page_size: int = 1000) -> Iterable:
    """
    Efficiently iterate a collection of any size, ordered by document name.
    Works across >10k docs without loading all at once.
    """
    last = None
    while True:
        q = coll_ref.order_by("__name__").limit(page_size)
        if last is not None:
            q = q.start_after(last)
        batch = list(q.stream())
        if not batch:
            break
        for d in batch:
            yield d
        last = batch[-1]


def export_collection(
    db: firestore.Client,
    collection_path: str,
    out_path: str,
    recursive: bool = False,
    page_size: int = 1000,
):
    """
    Export documents in collection_path to JSONL.
    Each line: {"_id": "...", "_path": "col/doc", "fields": {...}, "subcollections": {...?}}
    If recursive=True, subcollections are included inline under "subcollections".
    """
    coll_ref = db.collection(collection_path)

    with open(out_path, "w", encoding="utf-8") as f:
        for doc in paginate_collection(coll_ref, page_size=page_size):
            base = {
                "_id": doc.id,
                "_path": doc.reference.path,
                "fields": serialize_doc(doc.to_dict() or {}),
            }

            if recursive:
                subs = {}
                for subcoll in doc.reference.collections():
                    sub_name = subcoll.id
                    sub_docs = []
                    for sdoc in paginate_collection(subcoll, page_size=page_size):
                        sub_docs.append(
                            {
                                "_id": sdoc.id,
                                "_path": sdoc.reference.path,
                                "fields": serialize_doc(sdoc.to_dict() or {}),
                            }
                        )
                    subs[sub_name] = sub_docs
                base["subcollections"] = subs

            f.write(json.dumps(base, ensure_ascii=False))
            f.write("\n")


def get_client(credentials_path: Optional[str]) -> firestore.Client:
    if credentials_path:
        creds = service_account.Credentials.from_service_account_file(credentials_path)
        project_id = creds.project_id  # Prefer project from SA file
        return firestore.Client(project=project_id, credentials=creds)
    # Else rely on GOOGLE_APPLICATION_CREDENTIALS env var or default creds
    return firestore.Client()


def main():
    parser = argparse.ArgumentParser(description="Export a Firestore collection to JSONL.")
    parser.add_argument("--collection", required=True, help="Collection path (e.g., 'users' or 'tenants/acme/users').")
    parser.add_argument("--out", required=True, help="Output file path, e.g., ./export.jsonl")
    parser.add_argument("--recursive", action="store_true", help="Also export subcollections of each document.")
    parser.add_argument("--page-size", type=int, default=1000, help="Docs per page for iteration.")
    parser.add_argument("--credentials", help="Path to service account JSON (optional if env var is set).")
    args = parser.parse_args()

    db = get_client(args.credentials)
    export_collection(
        db=db,
        collection_path=args.collection,
        out_path=args.out,
        recursive=args.recursive,
        page_size=args.page_size,
    )
    print(f"Exported '{args.collection}' to {args.out}")


if __name__ == "__main__":
    main()
