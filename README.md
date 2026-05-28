## Research Question
How does geopolitical uncertainty affect the internationalization of SMEs?

## Theoretical Background
This project builds on the Uppsala Model of internationalization (Johanson & Vahlne, 1977), 
which argues that firms internationalize incrementally based on experiential knowledge. 
Under geopolitical uncertainty, smaller firms face higher relative costs and risks of 
foreign commitment, making them more likely to reduce foreign investment than larger firms.

## Hypotheses
- H1: Smaller firms reduce foreign investment more strongly under increased geopolitical 
uncertainty than larger firms.

## Variables

### Dependent variable (Y)
| Construct | Data Item(s) | Formula |
|-----------|-------------|---------|
| Capital Intensity | CAPX, AT | CAPX / AT |

### Independent variable (X)
| Construct | Data Item(s) | Formula |
|-----------|-------------|---------|
| Firm Size | AT | log(AT) |

### Controls
| Construct | Data Item(s) | Formula |
|-----------|-------------|---------|
| Leverage | DLTT, DLC, SEQ | (DLTT+DLC) / SEQ |
| Firm Age | FYEAR, ipodate | FYEAR - ipodate |
| RoA | NI, AT | NI / AT |
| Industry | SIC | categorical |
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