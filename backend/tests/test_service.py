import threading
import time
import unittest
from fastapi.testclient import TestClient
from flyarena_lab.service import create_app

TOKEN = "local-test-only-token"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


class ServiceTests(unittest.TestCase):
    def test_auth_bounds_lifecycle_and_race(self):
        entered, release = threading.Event(), threading.Event()
        def engine(options, progress, cancel):
            entered.set()
            release.wait(2)
            return {"finite": True}
        app = create_app(TOKEN, engine)
        with TestClient(app) as client:
            self.assertEqual(client.get('/api/v1/health').status_code, 200)
            self.assertEqual(client.post('/api/v1/jobs', json={}).status_code, 401)
            self.assertEqual(client.post('/api/v1/jobs', json={"ticks": 601}, headers=HEADERS).status_code, 422)
            self.assertEqual(client.post('/api/v1/jobs', content='x' * 4097, headers=HEADERS).status_code, 413)
            submitted = client.post('/api/v1/jobs', json={}, headers=HEADERS)
            self.assertEqual(submitted.status_code, 202)
            identifier = submitted.json()['id']
            self.assertTrue(entered.wait(2))
            self.assertEqual(client.post('/api/v1/jobs', json={}, headers=HEADERS).status_code, 409)
            self.assertEqual(client.delete(f'/api/v1/jobs/{identifier}', headers=HEADERS).json()['status'], 'cancelling')
            release.set()
            app.state.jobs.thread.join(3)
            terminal = client.get(f'/api/v1/jobs/{identifier}', headers=HEADERS).json()
            self.assertEqual(terminal['status'], 'cancelled')
            self.assertIsNone(terminal['result'])
            self.assertEqual(client.get('/api/v1/jobs/missing', headers=HEADERS).status_code, 404)

    def test_retention_completion_failure(self):
        app = create_app(TOKEN, lambda *args: {"done": True})
        with TestClient(app) as client:
            ids = []
            for _ in range(5):
                identifier = client.post('/api/v1/jobs', json={}, headers=HEADERS).json()['id']
                ids.append(identifier)
                app.state.jobs.thread.join(2)
            self.assertEqual(client.get(f'/api/v1/jobs/{ids[0]}', headers=HEADERS).status_code, 404)
            last = client.get(f'/api/v1/jobs/{ids[-1]}', headers=HEADERS).json()
            self.assertEqual(last['status'], 'completed')
            self.assertEqual(last['result'], {"done": True})
        def fail(*args):
            raise ValueError("private internal detail")
        app = create_app(TOKEN, fail)
        with TestClient(app) as client:
            identifier = client.post('/api/v1/jobs', json={}, headers=HEADERS).json()['id']
            app.state.jobs.thread.join(2)
            result = client.get(f'/api/v1/jobs/{identifier}', headers=HEADERS).json()
            self.assertEqual(result['status'], 'failed')
            self.assertNotIn('private', result['error'])
