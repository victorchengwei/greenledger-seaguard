# Deployment and operations

Both services run on **Cloud Run** in `asia-east1` (Taiwan). Commands assume Cloud Shell with a project selected.

## 1. Detector API

The weights are not stored in git (see `detector_api/README.md`). Put `best.pt` in `detector_api/`, then:

```bash
cd detector_api
gcloud run deploy greenledger-detector \
  --source . --region asia-east1 \
  --memory 2Gi --min-instances 0 --max-instances 1 \
  --allow-unauthenticated \
  --update-env-vars MODEL_PATH=best.pt,KEEP_CLASSES=all
```

Check it: `curl https://<detector-url>/health` returns `{"ok": true, "model": "best.pt"}`.

## 2. Dashboard

Store the Gemini key in Secret Manager instead of a plain variable:

```bash
printf '%s' "<your Gemini API key>" | gcloud secrets create gemini-api-key --data-file=-

gcloud run deploy greenledger-dashboard \
  --source . --region asia-east1 \
  --allow-unauthenticated \
  --set-env-vars DETECTOR_URL=https://<detector-url>,GEMINI_MODEL=<gemini model id> \
  --set-secrets GEMINI_API_KEY=gemini-api-key:latest
```

The Cloud Run service account needs the *Secret Manager Secret Accessor* role on that secret.

Keep the Cloud Shell tab open until `Service URL:` appears. A `--source` deploy builds the container in Cloud Build, then the CLI creates the new revision; if the session disconnects before the end, the build finishes but no new revision is created.

## 3. Operating it

| Topic | Practice |
|---|---|
| Cold starts | Services scale to zero. Before a demo or evaluation, `gcloud run services update <service> --region asia-east1 --min-instances 1`; set it back to 0 afterwards |
| Health | `GET /health` on the detector; Cloud Run restarts failed instances |
| Failures | The dashboard retries the detector once and shows a message if it is still unavailable; Gemini 429/503 errors are retried with back-off and then reported, not raised |
| Cost | Scale to zero, CPU-only inference, one batched Gemini call per image, and a billing budget with alerts |
| Rollback | `gcloud run services update-traffic <service> --region asia-east1 --to-revisions <previous-revision>=100` |
| Logs | Cloud Run → service → Logs, or `gcloud run services logs read <service> --region asia-east1` |
| Tests | `python -m pytest -q`; GitHub Actions runs the same suite on every push |

## 4. Scaling beyond the prototype

- Raise `--max-instances` for more concurrent users; both services are stateless.
- Move detection onto the ship (a small GPU) and send only detections and alerts ashore.
- Replace `sim/simulate.py` AIS with a live AIS feed (NMEA 0183 / IEC 61162 on board, or an AIS data provider ashore).
