from domain.model.portfolio_containers import PortfolioClientRaw, PortfolioClientsSim


def get_clients_sim_from_raw_portfolio_chunk(raw_chunk: PortfolioClientRaw) -> PortfolioClientsSim:
    return PortfolioClientsSim(raw_chunk.upb_vector, raw_chunk.sector_vector)