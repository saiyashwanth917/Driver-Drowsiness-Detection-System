import asyncio
import threading
import time

try:
    from winrt.windows.devices.geolocation import Geolocator
except ImportError:
    Geolocator = None


class WindowsLocation:
    def __init__(self, refresh_seconds=20):
        self.refresh_seconds = refresh_seconds
        self.lock = threading.Lock()
        self.data = None
        self.error = None
        self.running = False
        self.thread = None

    def start(self):
        if self.running or Geolocator is None:
            return

        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.running:
            try:
                result = asyncio.run(self._get_location())

                with self.lock:
                    self.data = result
                    self.error = None

            except Exception as e:
                with self.lock:
                    self.error = str(e)

            time.sleep(self.refresh_seconds)

    async def _get_location(self):
        geolocator = Geolocator()
        position = await geolocator.get_geoposition_async()
        coordinate = position.coordinate

        return {
            "latitude": coordinate.latitude,
            "longitude": coordinate.longitude,
            "accuracy": coordinate.accuracy,
            "timestamp": time.time(),
        }

    def get_latest(self, max_age=90):
        with self.lock:
            data = self.data.copy() if self.data else None

        if data is None:
            return None

        if time.time() - data["timestamp"] > max_age:
            return None

        return data

    def stop(self):
        self.running = False
