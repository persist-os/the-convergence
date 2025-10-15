"""
Plugin registry using pluggy for The Convergence framework.

Provides a robust plugin system similar to pytest's architecture.
"""

import pluggy
from typing import Any, Callable, Dict, List, Optional, Type
from contextlib import contextmanager

from convergence.core.protocols import (
    Plugin,
    LLMProvider,
    MABStrategy,
    MemorySystem,
    Agent,
)


# ============================================================================
# HOOK SPECIFICATIONS
# ============================================================================

hookspec = pluggy.HookspecMarker("convergence")
hookimpl = pluggy.HookimplMarker("convergence")


class ConvergenceHookSpec:
    """Hook specifications for The Convergence plugin system."""
    
    @hookspec
    def convergence_register_llm_provider(self) -> List[tuple[str, Type[LLMProvider]]]:
        """
        Register LLM provider implementations.
        
        Returns:
            List of (name, provider_class) tuples
        """
        pass
    
    @hookspec
    def convergence_register_mab_strategy(self) -> List[tuple[str, Type[MABStrategy]]]:
        """
        Register MAB strategy implementations.
        
        Returns:
            List of (name, strategy_class) tuples
        """
        pass
    
    @hookspec
    def convergence_register_memory_system(self) -> List[tuple[str, Type[MemorySystem]]]:
        """
        Register memory system implementations.
        
        Returns:
            List of (name, memory_class) tuples
        """
        pass
    
    @hookspec
    def convergence_register_plugin(self) -> List[Plugin]:
        """
        Register general plugins.
        
        Returns:
            List of plugin instances
        """
        pass


# ============================================================================
# PLUGIN REGISTRY
# ============================================================================

class PluginRegistry:
    """
    Central registry for all plugins in The Convergence.
    
    Uses pluggy for robust plugin management, similar to pytest.
    """
    
    def __init__(self) -> None:
        """Initialize the plugin manager."""
        self.pm = pluggy.PluginManager("convergence")
        self.pm.add_hookspecs(ConvergenceHookSpec)
        
        # Storage for registered components
        self._llm_providers: Dict[str, Type[LLMProvider]] = {}
        self._mab_strategies: Dict[str, Type[MABStrategy]] = {}
        self._memory_systems: Dict[str, Type[MemorySystem]] = {}
        self._plugins: Dict[str, Plugin] = {}
        
        # Register built-in plugins
        self._register_builtins()
    
    def _register_builtins(self) -> None:
        """Register built-in implementations."""
        # Built-in registrations happen here
        # Will be populated as we create plugins
        pass
    
    def register_plugin(self, plugin: Any, name: Optional[str] = None) -> None:
        """
        Register a plugin with the registry.
        
        Args:
            plugin: Plugin instance or class
            name: Optional name override
        """
        plugin_name = name or getattr(plugin, '__name__', str(plugin))
        self.pm.register(plugin, name=plugin_name)
        
        # Trigger hook to collect registrations
        self._collect_registrations()
    
    def unregister_plugin(self, plugin: Any) -> None:
        """Unregister a plugin."""
        self.pm.unregister(plugin)
        self._collect_registrations()
    
    def _collect_registrations(self) -> None:
        """Collect all registrations from plugins via hooks."""
        
        # Collect LLM providers
        for providers in self.pm.hook.convergence_register_llm_provider():
            for name, provider_class in providers:
                self._llm_providers[name] = provider_class
        
        # Collect MAB strategies
        for strategies in self.pm.hook.convergence_register_mab_strategy():
            for name, strategy_class in strategies:
                self._mab_strategies[name] = strategy_class
        
        # Collect memory systems
        for memory_systems in self.pm.hook.convergence_register_memory_system():
            for name, memory_class in memory_systems:
                self._memory_systems[name] = memory_class
        
        # Collect general plugins
        for plugins in self.pm.hook.convergence_register_plugin():
            for plugin in plugins:
                self._plugins[plugin.name] = plugin
    
    # LLM Provider Methods
    
    def register_llm_provider(
        self,
        name: str,
        provider_class: Type[LLMProvider]
    ) -> None:
        """Register an LLM provider."""
        self._llm_providers[name] = provider_class
    
    def get_llm_provider(
        self,
        name: str,
        **config: Any
    ) -> LLMProvider:
        """
        Get an LLM provider instance.
        
        Args:
            name: Provider name
            **config: Configuration for the provider
            
        Returns:
            Initialized LLM provider
            
        Raises:
            ValueError: If provider not found
        """
        if name not in self._llm_providers:
            available = ", ".join(self._llm_providers.keys())
            raise ValueError(
                f"LLM provider '{name}' not registered. "
                f"Available providers: {available}"
            )
        
        provider_class = self._llm_providers[name]
        return provider_class(**config)  # type: ignore
    
    def list_llm_providers(self) -> List[str]:
        """List all registered LLM providers."""
        return list(self._llm_providers.keys())
    
    # MAB Strategy Methods
    
    def register_mab_strategy(
        self,
        name: str,
        strategy_class: Type[MABStrategy]
    ) -> None:
        """Register a MAB strategy."""
        self._mab_strategies[name] = strategy_class
    
    def get_mab_strategy(
        self,
        name: str,
        **config: Any
    ) -> MABStrategy:
        """
        Get a MAB strategy instance.
        
        Args:
            name: Strategy name
            **config: Configuration for the strategy
            
        Returns:
            Initialized MAB strategy
            
        Raises:
            ValueError: If strategy not found
        """
        if name not in self._mab_strategies:
            available = ", ".join(self._mab_strategies.keys())
            raise ValueError(
                f"MAB strategy '{name}' not registered. "
                f"Available strategies: {available}"
            )
        
        strategy_class = self._mab_strategies[name]
        return strategy_class(**config)  # type: ignore
    
    def list_mab_strategies(self) -> List[str]:
        """List all registered MAB strategies."""
        return list(self._mab_strategies.keys())
    
    # Memory System Methods
    
    def register_memory_system(
        self,
        name: str,
        memory_class: Type[MemorySystem]
    ) -> None:
        """Register a memory system."""
        self._memory_systems[name] = memory_class
    
    def get_memory_system(
        self,
        name: str,
        **config: Any
    ) -> MemorySystem:
        """
        Get a memory system instance.
        
        Args:
            name: Memory system name
            **config: Configuration for the memory system
            
        Returns:
            Initialized memory system
            
        Raises:
            ValueError: If memory system not found
        """
        if name not in self._memory_systems:
            available = ", ".join(self._memory_systems.keys())
            raise ValueError(
                f"Memory system '{name}' not registered. "
                f"Available systems: {available}"
            )
        
        memory_class = self._memory_systems[name]
        return memory_class(**config)  # type: ignore
    
    def list_memory_systems(self) -> List[str]:
        """List all registered memory systems."""
        return list(self._memory_systems.keys())
    
    # General Plugin Methods
    
    def get_plugin(self, name: str) -> Plugin:
        """
        Get a plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance
            
        Raises:
            ValueError: If plugin not found
        """
        if name not in self._plugins:
            available = ", ".join(self._plugins.keys())
            raise ValueError(
                f"Plugin '{name}' not registered. "
                f"Available plugins: {available}"
            )
        
        return self._plugins[name]
    
    def list_plugins(self) -> List[str]:
        """List all registered plugins."""
        return list(self._plugins.keys())
    
    @contextmanager
    def temporary_plugin(self, plugin: Any, name: Optional[str] = None):
        """
        Temporarily register a plugin.
        
        Usage:
            with registry.temporary_plugin(MyPlugin()):
                # plugin is registered
                pass
            # plugin is unregistered
        """
        self.register_plugin(plugin, name)
        try:
            yield
        finally:
            self.unregister_plugin(plugin)


# ============================================================================
# GLOBAL REGISTRY INSTANCE
# ============================================================================

# Global registry instance - can be replaced for testing
_global_registry: Optional[PluginRegistry] = None


def get_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry


def reset_registry() -> None:
    """Reset the global registry (useful for testing)."""
    global _global_registry
    _global_registry = None

