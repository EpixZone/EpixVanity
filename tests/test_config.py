"""Tests for configuration management."""

import pytest
from epix_vanity.core.config import (
    EpixConfig, Bip44Config, Bech32Config, 
    Currency, FeeCurrency, GasPriceStep
)


class TestEpixConfig:
    """Test cases for EpixConfig class."""
    
    def test_default_testnet_config(self):
        """Test default testnet configuration."""
        config = EpixConfig.default_testnet()
        
        assert config.chain_id == "epix_1917-1"
        assert config.chain_name == "Epix"
        assert config.rpc == "https://rpc.testnet.epix.zone"
        assert config.rest == "https://api.testnet.epix.zone"
        assert config.bech32_config.bech32_prefix_acc_addr == "epix"
        assert config.bip44.coin_type == 60
    
    def test_get_address_prefix(self):
        """Test address prefix retrieval."""
        config = EpixConfig.default_testnet()
        prefix = config.get_address_prefix()
        assert prefix == "epix"
    
    def test_custom_config(self):
        """Test custom configuration creation."""
        custom_bech32 = Bech32Config(bech32_prefix_acc_addr="custom")
        config = EpixConfig(
            chain_id="custom_1-1",
            chain_name="Custom Chain",
            bech32_config=custom_bech32
        )
        
        assert config.chain_id == "custom_1-1"
        assert config.chain_name == "Custom Chain"
        assert config.get_address_prefix() == "custom"
    
    def test_from_dict(self):
        """Test configuration creation from dictionary."""
        config_dict = {
            "chainId": "test_1-1",
            "chainName": "Test Chain",
            "rpc": "https://rpc.test.com",
            "rest": "https://api.test.com",
            "bech32Config": {
                "bech32PrefixAccAddr": "test"
            }
        }
        
        config = EpixConfig.from_dict(config_dict)
        assert config.chain_id == "test_1-1"
        assert config.chain_name == "Test Chain"
        assert config.get_address_prefix() == "test"
    
    def test_validation_chain_id(self):
        """Test chain ID validation."""
        # Valid chain ID
        config = EpixConfig(chain_id="valid_1-1")
        assert config.chain_id == "valid_1-1"
        
        # Invalid chain ID
        with pytest.raises(ValueError):
            EpixConfig(chain_id="")
        
        with pytest.raises(ValueError):
            EpixConfig(chain_id=None)
    
    def test_validation_bech32_config(self):
        """Test bech32 configuration validation."""
        # Valid bech32 config
        valid_bech32 = Bech32Config(bech32_prefix_acc_addr="valid")
        config = EpixConfig(bech32_config=valid_bech32)
        assert config.bech32_config.bech32_prefix_acc_addr == "valid"
        
        # Invalid bech32 config
        with pytest.raises(ValueError):
            invalid_bech32 = Bech32Config(bech32_prefix_acc_addr="")
            EpixConfig(bech32_config=invalid_bech32)


class TestBip44Config:
    """Test cases for Bip44Config class."""
    
    def test_default_config(self):
        """Test default BIP44 configuration."""
        config = Bip44Config()
        assert config.coin_type == 60
    
    def test_custom_coin_type(self):
        """Test custom coin type."""
        config = Bip44Config(coin_type=118)
        assert config.coin_type == 118


class TestBech32Config:
    """Test cases for Bech32Config class."""
    
    def test_default_config(self):
        """Test default Bech32 configuration."""
        config = Bech32Config()
        assert config.bech32_prefix_acc_addr == "epix"
        assert config.bech32_prefix_acc_pub == "epixpub"
        assert config.bech32_prefix_val_addr == "epixvaloper"
    
    def test_custom_prefixes(self):
        """Test custom prefixes."""
        config = Bech32Config(
            bech32_prefix_acc_addr="custom",
            bech32_prefix_acc_pub="custompub"
        )
        assert config.bech32_prefix_acc_addr == "custom"
        assert config.bech32_prefix_acc_pub == "custompub"


class TestCurrency:
    """Test cases for Currency class."""
    
    def test_default_currency(self):
        """Test default currency configuration."""
        currency = Currency()
        assert currency.coin_denom == "EPIX"
        assert currency.coin_minimal_denom == "aepix"
        assert currency.coin_decimals == 18
        assert currency.coin_gecko_id == "unknown"
    
    def test_custom_currency(self):
        """Test custom currency configuration."""
        currency = Currency(
            coin_denom="CUSTOM",
            coin_minimal_denom="acustom",
            coin_decimals=6
        )
        assert currency.coin_denom == "CUSTOM"
        assert currency.coin_minimal_denom == "acustom"
        assert currency.coin_decimals == 6


class TestGasPriceStep:
    """Test cases for GasPriceStep class."""
    
    def test_default_gas_price(self):
        """Test default gas price configuration."""
        gas_price = GasPriceStep()
        assert gas_price.low == 0.01
        assert gas_price.average == 0.025
        assert gas_price.high == 0.04
    
    def test_custom_gas_price(self):
        """Test custom gas price configuration."""
        gas_price = GasPriceStep(
            low=0.001,
            average=0.01,
            high=0.1
        )
        assert gas_price.low == 0.001
        assert gas_price.average == 0.01
        assert gas_price.high == 0.1


class TestFeeCurrency:
    """Test cases for FeeCurrency class."""
    
    def test_default_fee_currency(self):
        """Test default fee currency configuration."""
        fee_currency = FeeCurrency()
        assert fee_currency.coin_denom == "EPIX"
        assert fee_currency.gas_price_step.low == 0.01
        assert fee_currency.gas_price_step.average == 0.025
        assert fee_currency.gas_price_step.high == 0.04
    
    def test_custom_fee_currency(self):
        """Test custom fee currency configuration."""
        custom_gas_price = GasPriceStep(low=0.005, average=0.015, high=0.025)
        fee_currency = FeeCurrency(
            coin_denom="CUSTOM",
            gas_price_step=custom_gas_price
        )
        assert fee_currency.coin_denom == "CUSTOM"
        assert fee_currency.gas_price_step.low == 0.005
