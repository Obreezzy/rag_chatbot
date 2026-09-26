"""
Run this ONCE locally (or whenever your policy PDFs change) to pre-build
the search index and save it as index_cache.pkl.

    python build_cache.py

Commit the resulting index_cache.pkl to your repo. From then on, build_index()
in rag_system.py will load this cache instantly at startup instead of
re-parsing every PDF and rebuilding BM25 from scratch every single deploy.

Delete index_cache.pkl and re-run this script whenever you add, remove,
or edit a policy document.
"""

from rag_system import build_index, index_ready

if __name__ == "__main__":
    build_index()
    if index_ready.is_set():
        print("\nDone. Commit index_cache.pkl to your repo and redeploy.")
    else:
        print("\nSomething went wrong — index was not built. Check the errors above.")
