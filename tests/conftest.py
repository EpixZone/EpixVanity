"""Pytest configuration and fixtures."""

import pytest
from epix_vanity.core.config import EpixConfig
from epix_vanity.core.crypto import EpixCrypto
from epix_vanity.utils.patterns import PatternValidator


@pytest.fixture
def epix_config():
    """Provide a default Epix configuration for tests."""
    return EpixConfig.default_testnet()


@pytest.fixture
def epix_crypto(epix_config):
    """Provide an EpixCrypto instance for tests."""
    return EpixCrypto(epix_config)


@pytest.fixture
def pattern_validator(epix_config):
    """Provide a PatternValidator instance for tests."""
    return PatternValidator(epix_config.get_address_prefix())


@pytest.fixture
def sample_private_key():
    """Provide a sample private key for deterministic tests."""
    return bytes.fromhex("1234567890123456789012345678901234567890123456789012345678901234")


@pytest.fixture
def sample_keypair(epix_crypto, sample_private_key):
    """Provide a sample keypair for tests."""
    public_key = epix_crypto.private_key_to_public_key(sample_private_key)
    eth_address = epix_crypto.public_key_to_eth_address(public_key)
    bech32_address = epix_crypto.eth_address_to_bech32(eth_address)
    
    from epix_vanity.core.crypto import KeyPair
    return KeyPair(
        private_key=sample_private_key,
        public_key=public_key,
        address=bech32_address,
        eth_address=eth_address
    )
