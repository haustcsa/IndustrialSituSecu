"""
Modbus-TCP Protocol Implementation
"""
import struct
from typing import Dict, Any, List
from ..base import ProtocolInterface


class ModbusTCP(ProtocolInterface):
    """
    Modbus-TCP Protocol Implementation
    """

    FUNCTION_CODES = {
        0x01: "Read Coils",
        0x02: "Read Discrete Inputs",
        0x03: "Read Holding Registers",
        0x04: "Read Input Registers",
        0x05: "Write Single Coil",
        0x06: "Write Single Register",
        0x0F: "Write Multiple Coils",
        0x10: "Write Multiple Registers",
        0x17: "Read/Write Multiple Registers"
    }

    def __init__(self):
        self.transaction_id = 0

    def generate_valid_message(
            self,
            function_code: int = 0x03,
            start_address: int = 0,
            quantity: int = 10,
            unit_id: int = 1
    ) -> bytes:
        """Generate valid Modbus-TCP message"""

        self.transaction_id = (self.transaction_id + 1) % 65536

        # MBAP Header
        transaction_id = struct.pack('>H', self.transaction_id)
        protocol_id = struct.pack('>H', 0)  # Always 0 for Modbus
        unit_id_byte = struct.pack('B', unit_id)

        # PDU (Protocol Data Unit)
        if function_code in [0x01, 0x02, 0x03, 0x04]:
            # Read functions
            pdu = struct.pack('>BHH', function_code, start_address, quantity)
        elif function_code in [0x05, 0x06]:
            # Write single
            pdu = struct.pack('>BHH', function_code, start_address, quantity)
        elif function_code in [0x0F, 0x10]:
            # Write multiple
            byte_count = 2 * quantity
            data = b'\x00\x0A' * quantity  # Dummy data
            pdu = struct.pack('>BHHB', function_code, start_address, quantity, byte_count) + data
        else:
            pdu = struct.pack('B', function_code)

        length = struct.pack('>H', len(pdu) + 1)  # +1 for unit ID

        message = transaction_id + protocol_id + length + unit_id_byte + pdu
        return message

    def parse_message(self, message: bytes) -> Dict[str, Any]:
        """Parse Modbus-TCP message"""
        if len(message) < 8:
            return {'valid': False, 'error': 'Message too short'}

        try:
            parsed = {
                'transaction_id': struct.unpack('>H', message[0:2])[0],
                'protocol_id': struct.unpack('>H', message[2:4])[0],
                'length': struct.unpack('>H', message[4:6])[0],
                'unit_id': message[6],
                'function_code': message[7],
                'data': message[8:],
                'valid': True
            }

            parsed['function_name'] = self.FUNCTION_CODES.get(
                parsed['function_code'],
                f"Unknown (0x{parsed['function_code']: 02X})"
            )

            return parsed
        except Exception as e:
            return {'valid': False, 'error': str(e)}

    def validate_message(self, message: bytes) -> bool:
        """Validate Modbus-TCP message"""
        if len(message) < 8:
            return False

        try:
            protocol_id = struct.unpack('>H', message[2:4])[0]
            if protocol_id != 0:
                return False

            length = struct.unpack('>H', message[4:6])[0]
            if length != len(message) - 6:
                return False

            function_code = message[7]
            if function_code not in self.FUNCTION_CODES:
                return False

            return True
        except:
            return False

    def mutate_message(self, message: bytes) -> bytes:
        """Apply random mutation to message"""
        import random

        if len(message) == 0:
            return message

        message_list = list(message)
        mutation_type = random.choice(['flip_bit', 'change_byte', 'insert', 'delete'])

        if mutation_type == 'flip_bit':
            pos = random.randint(0, len(message_list) - 1)
            bit = random.randint(0, 7)
            message_list[pos] ^= (1 << bit)
        elif mutation_type == 'change_byte':
            pos = random.randint(0, len(message_list) - 1)
            message_list[pos] = random.randint(0, 255)
        elif mutation_type == 'insert':
            pos = random.randint(0, len(message_list))
            message_list.insert(pos, random.randint(0, 255))
        elif mutation_type == 'delete' and len(message_list) > 1:
            pos = random.randint(0, len(message_list) - 1)
            del message_list[pos]

        return bytes(message_list)

    def get_function_codes(self) -> List[int]:
        """Return list of valid function codes"""
        return list(self.FUNCTION_CODES.keys())