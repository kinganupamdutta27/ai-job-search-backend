import httpx, json

r = httpx.get("http://localhost:8001/api/workflow/f2d845b0-2a86-4ac2-abb1-baa27791ce8a/status")
d = r.json()

print("=== STATUS ===")
print(f"status: {d['status']}")
print(f"step: {d['current_step']}")
print(f"jobs_found: {d['jobs_found']}")
print(f"emails_generated: {d.get('emails_generated', 0)}")
print(f"errors: {d.get('errors', [])}")

print("\n=== JOB LISTINGS ===")
jobs = d.get("job_listings", [])
for i, j in enumerate(jobs):
    print(f"\n  [{i}] {j.get('title')} @ {j.get('company')}")
    print(f"      URL: {j.get('url', 'N/A')[:80]}")
    print(f"      Source: {j.get('source')}")
    hrs = j.get("hr_contacts", [])
    print(f"      HR contacts: {len(hrs)}")
    for h in hrs:
        print(f"        - {h.get('name')} <{h.get('email')}> ({h.get('source')})")

print("\n=== DRAFT EMAILS ===")
drafts = d.get("draft_emails", [])
print(f"Count: {len(drafts)}")
for de in drafts:
    print(f"  To: {de.get('to_email')} | {de.get('subject', '')[:60]}")
