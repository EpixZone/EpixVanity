"""Tests for cryptographic functions."""

import pytest
from epix_vanity.core.crypto import EpixCrypto, KeyPair
from epix_vanity.core.config import EpixConfig


class TestEpixCrypto:
    """Test cases for EpixCrypto class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.config = EpixConfig.default_testnet()
        self.crypto = EpixCrypto(self.config)
    
    def test_generate_private_key(self):
        """Test private key generation."""
        private_key = self.crypto.generate_private_key()
        
        assert isinstance(private_key, bytes)
        assert len(private_key) == 32
        
        # Generate another key and ensure they're different
        private_key2 = self.crypto.generate_private_key()
        assert private_key != private_key2
    
    def test_private_key_to_public_key(self):
        """Test public key derivation from private key."""
        private_key = self.crypto.generate_private_key()
        public_key = self.crypto.private_key_to_public_key(private_key)
        
        assert isinstance(public_key, bytes)
        assert len(public_key) == 65
        assert public_key[0] == 0x04  # Uncompressed public key prefix
    
    def test_public_key_to_eth_address(self):
        """Test Ethereum address derivation from public key."""
        private_key = self.crypto.generate_private_key()
        public_key = self.crypto.private_key_to_public_key(private_key)
        eth_address = self.crypto.public_key_to_eth_address(public_key)
        
        assert isinstance(eth_address, str)
        assert eth_address.startswith("0x")
        assert len(eth_address) == 42  # 0x + 40 hex characters
        
        # Verify it's valid hex
        int(eth_address[2:], 16)
    
    def test_eth_address_to_bech32(self):
        """Test Bech32 address conversion."""
        eth_address = "0x1234567890123456789012345678901234567890"
        bech32_address = self.crypto.eth_address_to_bech32(eth_address)
        
        assert isinstance(bech32_address, str)
        assert bech32_address.startswith(self.config.get_address_prefix())
    
    def test_generate_keypair(self):
        """Test complete keypair generation."""
        keypair = self.crypto.generate_keypair()
        
        assert isinstance(keypair, KeyPair)
        assert isinstance(keypair.private_key, bytes)
        assert isinstance(keypair.public_key, bytes)
        assert isinstance(keypair.address, str)
        assert isinstance(keypair.eth_address, str)
        
        assert len(keypair.private_key) == 32
        assert len(keypair.public_key) == 65
        assert keypair.address.startswith(self.config.get_address_prefix())
        assert keypair.eth_address.startswith("0x")
    
    def test_validate_private_key(self):
        """Test private key validation."""
        # Valid private key
        valid_key = self.crypto.generate_private_key()
        assert self.crypto.validate_private_key(valid_key)
        
        # Invalid length
        assert not self.crypto.validate_private_key(b"short")
        assert not self.crypto.validate_private_key(b"x" * 33)
        
        # Zero key (invalid)
        zero_key = b"\x00" * 32
        assert not self.crypto.validate_private_key(zero_key)
    
    def test_private_key_hex_conversion(self):
        """Test private key hex string conversion."""
        private_key = self.crypto.generate_private_key()
        
        # Convert to hex
        hex_string = self.crypto.private_key_to_hex(private_key)
        assert hex_string.startswith("0x")
        assert len(hex_string) == 66  # 0x + 64 hex characters
        
        # Convert back from hex
        recovered_key = self.crypto.private_key_from_hex(hex_string)
        assert recovered_key == private_key
        
        # Test without 0x prefix
        hex_no_prefix = hex_string[2:]
        recovered_key2 = self.crypto.private_key_from_hex(hex_no_prefix)
        assert recovered_key2 == private_key
    
    def test_deterministic_generation(self):
        """Test that same private key produces same results."""
        # Use a fixed private key for deterministic testing
        private_key = bytes.fromhex("1234567890123456789012345678901234567890123456789012345678901234")
        
        # Generate multiple times
        public_key1 = self.crypto.private_key_to_public_key(private_key)
        public_key2 = self.crypto.private_key_to_public_key(private_key)
        assert public_key1 == public_key2
        
        eth_address1 = self.crypto.public_key_to_eth_address(public_key1)
        eth_address2 = self.crypto.public_key_to_eth_address(public_key2)
        assert eth_address1 == eth_address2
        
        bech32_address1 = self.crypto.eth_address_to_bech32(eth_address1)
        bech32_address2 = self.crypto.eth_address_to_bech32(eth_address2)
        assert bech32_address1 == bech32_address2
    
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        # Invalid private key length
        with pytest.raises(ValueError):
            self.crypto.private_key_to_public_key(b"short")
        
        # Invalid public key
        with pytest.raises(ValueError):
            self.crypto.public_key_to_eth_address(b"invalid")
        
        # Invalid Ethereum address
        with pytest.raises(ValueError):
            self.crypto.eth_address_to_bech32("invalid")
        
        # Invalid hex string
        with pytest.raises(ValueError):
            self.crypto.private_key_from_hex("invalid_hex")
        
        with pytest.raises(ValueError):
            self.crypto.private_key_from_hex("0x123")  # Too short
