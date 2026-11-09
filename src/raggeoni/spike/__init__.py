"""Spike Investigator: explain NEM price spikes with a team of agents.

Data comes from AEMO NEMWeb (prices, demand, interconnectors, forecasts,
market notices) and Open Electricity (generation by fuel type). Analysis is
done by typed, tested tools; agents decide what to run and how to read it.
Every number in a final report is checked against the tool outputs that
produced it.
"""

__all__ = ["__version__"]
__version__ = "0.2.0"
