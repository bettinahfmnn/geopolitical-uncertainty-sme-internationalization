## Research Question
How does geopolitical uncertainty affect the capital investment behavior of SMEs compared to large firms in Europe?

## Theoretical Background
This project builds on the Uppsala Model of internationalization (Johanson & Vahlne, 1977), which argues that firms internationalize incrementally based on experiential knowledge. Under geopolitical uncertainty, smaller firms face higher relative costs and risks of foreign commitment due to limited resources and risk-bearing capacity (Lu & Beamish, 2001). The geopolitical risk index (GPR) by Caldara & Iacoviello (2022) is used as an external measure of geopolitical uncertainty.

## Hypotheses
- H1: SMEs reduce capital investment more strongly in response to increases in geopolitical risk than large firms.

## SME Definition
Following the EU definition, a firm is classified as an SME if it has fewer than 250 employees (emp < 0.25 in Compustat, where emp is measured in thousands) or total assets ≤ €43 million.

## Sample & Country Focus
- Geography: European firms (AUT, BEL, CHE, CZE, DEU, DNK, ESP, FIN, FRA, GBR, GRC, HUN, IRL, ITA, LUX, NLD, NOR, POL, PRT, ROU, SWE, SVK, SVN)
- Time period: Fiscal years 2015–2024
- Source: WRDS Compustat Global (comp_global_daily.g_funda)

## Variables

### Dependent variable (Y)
| Construct | Data Item(s) | Formula |
|-----------|-------------|---------|
| Capital Intensity | CAPX, AT | CAPX / AT |

### Independent variable (X)
| Construct | Data Item(s) | Formula |
|-----------|-------------|---------|
| Firm Size | AT | log(AT) |
| Geopolitical Risk | GPR Index | External (Caldara & Iacoviello, 2022) |

### Controls
| Construct | Data Item(s) | Formula |
|-----------|-------------|---------|
| Leverage | DLTT, DLC, SEQ | (DLTT+DLC) / SEQ |
| Firm Age | FYEAR | proxy variable |
| RoA | NICON, AT | NICON / AT |
| Industry | SICH | categorical |

## Data
| Item | Detail |
|------|--------|
| Source | WRDS / Compustat Global |
| Table | comp_global_daily.g_funda |
| Downloaded | 2026-05-28 |
| License | WRDS subscriber agreement |
| Fiscal years | 2015-2024 |
| Raw rows | 338,462 |
| Clean rows | 89,713 |