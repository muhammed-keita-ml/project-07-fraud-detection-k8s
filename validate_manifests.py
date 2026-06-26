import yaml
import sys

files = ["k8s/deployment.yaml", "k8s/manifests.yaml"]

for f in files:
    try:
        docs = list(yaml.safe_load_all(open(f)))
        print(f"OK: {f} ({len(docs)} document(s))")
        for doc in docs:
            if doc:
                kind = doc["kind"]
                name = doc["metadata"]["name"]
                print(f"  - {kind}: {name}")
    except Exception as e:
        print(f"ERROR: {f}: {e}")
        sys.exit(1)

print("\nAll manifests valid.")
