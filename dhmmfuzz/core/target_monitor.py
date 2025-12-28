"""
Target Monitoring and Feedback Collection
"""
import socket
import time
import hashlib
from typing import Dict, Optional, Tuple
import logging


class TargetMonitor:
    """
    Monitor target ICS for crashes and collect feedback
    """

    def __init__(
            self,
            host: str,
            port: int,
            timeout: float = 2.0,
            max_retries: int = 3
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_retries = max_retries
        self.logger = logging.getLogger("TargetMonitor")

    def send_and_monitor(self, message: bytes) -> Dict:
        """
        Send message and monitor for crashes/anomalies
        """
        result = {
            'response_received': False,
            'response_data': None,
            'response_time': None,
            'crash_detected': False,
            'crash_signature': None,
            'connection_error': False,
            'timeout': False
        }

        start_time = time.time()

        try:
            # Create socket connection
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout)
                sock.connect((self.host, self.port))

                # Send message
                sock.sendall(message)

                # Receive response
                try:
                    response = sock.recv(4096)
                    result['response_received'] = True
                    result['response_data'] = response
                    result['response_time'] = time.time() - start_time

                    # Check for error responses
                    if self._is_error_response(response):
                        result['crash_detected'] = self._analyze_error(response)
                        result['crash_signature'] = hashlib.md5(message).hexdigest()

                except socket.timeout:
                    result['timeout'] = True
                    # Timeout might indicate crash
                    if self._verify_target_alive():
                        result['crash_detected'] = False
                    else:
                        result['crash_detected'] = True
                        result['crash_signature'] = hashlib.md5(message).hexdigest()

        except ConnectionRefusedError:
            result['connection_error'] = True
            result['crash_detected'] = True
            result['crash_signature'] = hashlib.md5(message).hexdigest()
        except Exception as e:
            self.logger.error(f"Error monitoring target: {e}")
            result['connection_error'] = True

        return result

    def _is_error_response(self, response: bytes) -> bool:
        """Check if response indicates an error"""
        if len(response) < 8:
            return False

        # Modbus exception response has MSB set in function code
        function_code = response[7] if len(response) > 7 else 0
        return (function_code & 0x80) != 0

    def _analyze_error(self, response: bytes) -> bool:
        """
        Analyze error response to determine if it's a crash
        """
        if len(response) < 9:
            return True  # Malformed response

        exception_code = response[8] if len(response) > 8 else 0

        # Modbus exception codes 1-4 are normal, others might indicate issues
        critical_exceptions = [0x05, 0x06, 0x0A, 0x0B]

        return exception_code in critical_exceptions

    def _verify_target_alive(self) -> bool:
        """Verify if target is still responsive"""
        for _ in range(self.max_retries):
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1.0)
                    sock.connect((self.host, self.port))
                    return True
            except:
                time.sleep(0.5)
        return False