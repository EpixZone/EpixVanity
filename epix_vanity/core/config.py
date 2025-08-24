"""Configuration management for Epix blockchain parameters."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator


class Bip44Config(BaseModel):
    """BIP44 configuration for Epix."""
    coin_type: int = Field(60, description="BIP44 coin type")


class Bech32Config(BaseModel):
    """Bech32 configuration for Epix addresses."""
    bech32_prefix_acc_addr: str = Field("epix", description="Account address prefix")
    bech32_prefix_acc_pub: str = Field("epixpub", description="Account public key prefix")
    bech32_prefix_val_addr: str = Field("epixvaloper", description="Validator address prefix")
    bech32_prefix_val_pub: str = Field("epixvaloperpub", description="Validator public key prefix")
    bech32_prefix_cons_addr: str = Field("epixvalcons", description="Consensus address prefix")
    bech32_prefix_cons_pub: str = Field("epixvalconspub", description="Consensus public key prefix")


class Currency(BaseModel):
    """Currency configuration."""
    coin_denom: str = Field("EPIX", description="Coin denomination")
    coin_minimal_denom: str = Field("aepix", description="Minimal coin denomination")
    coin_decimals: int = Field(18, description="Number of decimal places")
    coin_gecko_id: str = Field("unknown", description="CoinGecko ID")
    coin_image_url: Optional[str] = Field(None, description="Coin image URL")


class GasPriceStep(BaseModel):
    """Gas price step configuration."""
    low: float = Field(0.01, description="Low gas price")
    average: float = Field(0.025, description="Average gas price")
    high: float = Field(0.04, description="High gas price")


class FeeCurrency(Currency):
    """Fee currency configuration with gas price steps."""
    gas_price_step: GasPriceStep = Field(default_factory=GasPriceStep)


class EpixConfig(BaseModel):
    """Complete Epix blockchain configuration."""
    
    chain_id: str = Field("epix_1917-1", description="Chain ID")
    chain_name: str = Field("Epix", description="Chain name")
    rpc: str = Field("https://rpc.testnet.epix.zone", description="RPC endpoint")
    rest: str = Field("https://api.testnet.epix.zone", description="REST API endpoint")
    chain_symbol_image_url: Optional[str] = Field(None, description="Chain symbol image URL")
    
    bip44: Bip44Config = Field(default_factory=Bip44Config)
    bech32_config: Bech32Config = Field(default_factory=Bech32Config)
    
    currencies: list[Currency] = Field(default_factory=lambda: [Currency()])
    fee_currencies: list[FeeCurrency] = Field(default_factory=lambda: [FeeCurrency()])
    stake_currency: Currency = Field(default_factory=Currency)
    
    features: list[str] = Field(
        default_factory=lambda: ["eth-address-gen", "eth-key-sign", "ibc-transfer", "ibc-go"],
        description="Supported features"
    )
    
    @validator('chain_id')
    def validate_chain_id(cls, v: str) -> str:
        """Validate chain ID format."""
        if not v or not isinstance(v, str):
            raise ValueError("Chain ID must be a non-empty string")
        return v
    
    @validator('bech32_config')
    def validate_bech32_config(cls, v: Bech32Config) -> Bech32Config:
        """Validate bech32 configuration."""
        if not v.bech32_prefix_acc_addr:
            raise ValueError("Account address prefix cannot be empty")
        return v
    
    @classmethod
    def default_testnet(cls) -> "EpixConfig":
        """Create default testnet configuration."""
        return cls(
            chain_id="epix_1917-1",
            chain_name="Epix",
            rpc="https://rpc.testnet.epix.zone",
            rest="https://api.testnet.epix.zone",
            chain_symbol_image_url="https://raw.githubusercontent.com/EpixZone/assets/refs/heads/main/images/icons/icon.png"
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "EpixConfig":
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def get_address_prefix(self) -> str:
        """Get the address prefix for vanity generation."""
        return self.bech32_config.bech32_prefix_acc_addr
