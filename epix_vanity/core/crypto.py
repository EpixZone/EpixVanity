"""Core cryptographic functions for Epix vanity address generation."""

import os
import hashlib
from typing import Tuple, Optional
from dataclasses import dataclass

import ecdsa
from Crypto.Hash import keccak
import bech32

from .config import EpixConfig


@dataclass
class KeyPair:
    """Represents a cryptographic key pair."""
    private_key: bytes
    public_key: bytes
    address: str
    eth_address: str


class EpixCrypto:
    """Core cryptographic operations for Epix blockchain."""
    
    def __init__(self, config: Optional[EpixConfig] = None):
        """Initialize with Epix configuration."""
        self.config = config or EpixConfig.default_testnet()
        self.address_prefix = self.config.get_address_prefix()
    
    def generate_private_key(self) -> bytes:
        """Generate a random 32-byte private key."""
        return os.urandom(32)
    
    def private_key_to_public_key(self, private_key: bytes) -> bytes:
        """Convert private key to uncompressed public key."""
        if len(private_key) != 32:
            raise ValueError("Private key must be 32 bytes")
        
        # Create signing key from private key
        signing_key = ecdsa.SigningKey.from_string(
            private_key, 
            curve=ecdsa.SECP256k1
        )
        
        # Get uncompressed public key (65 bytes: 0x04 + 32 bytes x + 32 bytes y)
        verifying_key = signing_key.get_verifying_key()
        public_key = b'\x04' + verifying_key.to_string()
        
        return public_key
    
    def public_key_to_eth_address(self, public_key: bytes) -> str:
        """Convert public key to Ethereum-style address."""
        if len(public_key) != 65 or public_key[0] != 0x04:
            raise ValueError("Public key must be 65 bytes starting with 0x04")
        
        # Remove the 0x04 prefix and hash the remaining 64 bytes
        public_key_hash = keccak.new(digest_bits=256)
        public_key_hash.update(public_key[1:])
        
        # Take the last 20 bytes as the Ethereum address
        eth_address_bytes = public_key_hash.digest()[-20:]
        
        # Convert to hex string with 0x prefix
        return "0x" + eth_address_bytes.hex()
    
    def eth_address_to_bech32(self, eth_address: str) -> str:
        """Convert Ethereum address to Bech32 format."""
        if not eth_address.startswith("0x") or len(eth_address) != 42:
            raise ValueError("Invalid Ethereum address format")
        
        # Remove 0x prefix and convert to bytes
        address_bytes = bytes.fromhex(eth_address[2:])
        
        # Convert to 5-bit groups for bech32 encoding
        converted = bech32.convertbits(address_bytes, 8, 5)
        if converted is None:
            raise ValueError("Failed to convert address to bech32 format")
        
        # Encode with the address prefix
        bech32_address = bech32.bech32_encode(self.address_prefix, converted)
        if bech32_address is None:
            raise ValueError("Failed to encode bech32 address")
        
        return bech32_address
    
    def generate_keypair(self) -> KeyPair:
        """Generate a complete key pair with addresses."""
        private_key = self.generate_private_key()
        public_key = self.private_key_to_public_key(private_key)
        eth_address = self.public_key_to_eth_address(public_key)
        bech32_address = self.eth_address_to_bech32(eth_address)
        
        return KeyPair(
            private_key=private_key,
            public_key=public_key,
            address=bech32_address,
            eth_address=eth_address
        )
    
    def validate_private_key(self, private_key: bytes) -> bool:
        """Validate that a private key is valid for secp256k1."""
        if len(private_key) != 32:
            return False
        
        # Check that private key is in valid range (1 to n-1 where n is curve order)
        private_key_int = int.from_bytes(private_key, 'big')
        curve_order = ecdsa.SECP256k1.order
        
        return 1 <= private_key_int < curve_order
    
    def private_key_to_wif(self, private_key: bytes, compressed: bool = True) -> str:
        """Convert private key to Wallet Import Format (WIF)."""
        if not self.validate_private_key(private_key):
            raise ValueError("Invalid private key")
        
        # Add version byte (0x80 for mainnet)
        extended_key = b'\x80' + private_key
        
        # Add compression flag if compressed
        if compressed:
            extended_key += b'\x01'
        
        # Double SHA256 hash for checksum
        hash1 = hashlib.sha256(extended_key).digest()
        hash2 = hashlib.sha256(hash1).digest()
        checksum = hash2[:4]
        
        # Combine and encode in base58
        full_key = extended_key + checksum
        
        # Simple base58 encoding (for production, use a proper base58 library)
        alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
        num = int.from_bytes(full_key, 'big')
        
        if num == 0:
            return alphabet[0]
        
        result = ""
        while num > 0:
            num, remainder = divmod(num, 58)
            result = alphabet[remainder] + result
        
        # Add leading zeros
        for byte in full_key:
            if byte == 0:
                result = alphabet[0] + result
            else:
                break
        
        return result
    
    @staticmethod
    def private_key_from_hex(hex_string: str) -> bytes:
        """Convert hex string to private key bytes."""
        if hex_string.startswith("0x"):
            hex_string = hex_string[2:]
        
        if len(hex_string) != 64:
            raise ValueError("Private key hex string must be 64 characters (32 bytes)")
        
        return bytes.fromhex(hex_string)
    
    @staticmethod
    def private_key_to_hex(private_key: bytes) -> str:
        """Convert private key bytes to hex string."""
        return "0x" + private_key.hex()
