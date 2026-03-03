"""Full E2E test with HITL approval using user's real resume."""

import httpx
import time
import sys

sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://localhost:8001"
CV_PATH = r"C:\Users\anupa\Downloads\Project-Personal\MalePOC\Dutta_Anupam_Resume_01022026.pdf"


def run():
    print("=" * 60)
    print("🚀 FULL E2E TEST — Including HITL Approval")
    print("=" * 60)

    with httpx.Client(timeout=30) as c:
        # 1. Health
        print("\n[1/6] Health check...")
        r = c.get(f"{BASE}/health")
        h = r.json()
        print(f"   ✅ Backend healthy | OpenAI={h['openai_configured']} | Tavily={h['tavily_configured']} | SMTP={h['smtp_configured']}")

        # 2. Upload real PDF
        print("\n[2/6] Uploading real resume PDF...")
        with open(CV_PATH, "rb") as f:
            r = c.post(f"{BASE}/api/cv/upload", files={"file": ("resume.pdf", f, "application/pdf")}, timeout=60)
        if r.status_code != 200:
            print(f"   ❌ Upload failed: {r.text}")
            return
        up = r.json()
        print(f"   ✅ Uploaded: {up.get('text_length')} chars extracted from PDF")

        # 3. Start workflow
        print("\n[3/6] Starting LangGraph workflow...")
        payload = {
            "cv_file_path": up["file_path"],
            "search_location": "India Remote",
            "max_jobs": 3,
        }
        r = c.post(f"{BASE}/api/workflow/start", json=payload, timeout=120)
        wf = r.json()
        run_id = wf.get("run_id")
        print(f"   ✅ Workflow started: {run_id}")
        print(f"   Status: {wf.get('status')} | Jobs: {wf.get('jobs_found')} | Emails: {wf.get('emails_generated')}")

        if wf.get("status") == "failed":
            print(f"   ❌ Workflow failed immediately: {wf.get('errors')}")
            return

        # 4. Poll until awaiting_review
        print("\n[4/6] Polling workflow status...")
        status = wf.get("status")
        if status != "awaiting_review":
            for i in range(60):
                time.sleep(3)
                r = c.get(f"{BASE}/api/workflow/{run_id}/status")
                st = r.json()
                status = st.get("status")
                step = st.get("current_step")
                jobs = st.get("jobs_found", 0)
                emails = len(st.get("draft_emails", []))
                print(f"   ⏳ [{i:02d}] Step={str(step):<22} Status={str(status):<16} Jobs={jobs} Emails={emails}")
                if status in ("awaiting_review", "completed", "failed"):
                    break
            else:
                print("   ❌ Timed out waiting for workflow")
                return

        if status == "failed":
            r = c.get(f"{BASE}/api/workflow/{run_id}/status")
            print(f"   ❌ Failed: {r.json().get('errors')}")
            return

        # 5. HITL Approval
        print("\n[5/6] 🧑‍💼 HITL: Reviewing and approving emails...")
        r = c.get(f"{BASE}/api/workflow/{run_id}/status")
        st = r.json()
        drafts = st.get("draft_emails", [])
        print(f"   Found {len(drafts)} draft emails to review:")
        for d in drafts:
            print(f"   📧 To: {d.get('to_name', '?')} <{d.get('to_email', '?')}> | {d.get('job_title', '?')} @ {d.get('company', '?')}")

        decisions = [{"email_id": d["id"], "approved": True} for d in drafts]
        print(f"\n   ✅ Approving all {len(decisions)} emails...")
        r = c.post(
            f"{BASE}/api/workflow/{run_id}/review",
            json={"decisions": decisions},
            timeout=120,
        )
        review = r.json()
        print(f"   Result: status={review.get('status')} | approved={review.get('approved_count')} | rejected={review.get('rejected_count')}")

        if review.get("sent_results"):
            print("\n[6/6] 📤 Email send results:")
            for sr in review["sent_results"]:
                icon = "✅" if sr.get("success") else "❌"
                print(f"   {icon} {sr.get('email_id', '?')[:8]}... → {'Sent' if sr.get('success') else sr.get('error', 'Failed')}")
        else:
            print("\n[6/6] No send results returned (emails may not have been sent)")

        if review.get("errors"):
            print(f"\n   ⚠️ Errors: {review['errors']}")

    print("\n" + "=" * 60)
    print("🎉 FULL E2E TEST WITH HITL COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run()
