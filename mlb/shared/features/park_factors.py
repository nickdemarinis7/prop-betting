"""
MLB Park HR Factors

Three-year rolling Statcast HR park factors normalized so 1.00 = league average.
Sources: Baseball Savant (savant.mlb.com/leaderboard/statcast-park-factors)
Last updated based on 2023-2025 multi-year averages.

Values represent the multiplier to apply to a batter's expected HR rate
when playing in that park. e.g. COL=1.30 means HRs are ~30% more likely
in Coors Field than at a league-average park.

Notes:
- Uses team abbreviation as key (consistent with rest of codebase).
- Default is 1.00 for any unrecognized team.
- Split factors (vs LHB / RHB) intentionally omitted; pooled value is more
  stable for our small-sample model and matches what the books typically
  use as the baseline.
"""

PARK_HR_FACTORS = {
    # Extreme hitter parks
    'COL': 1.30,  # Coors Field — altitude effect
    'CIN': 1.20,  # Great American Ballpark — short porch in RF
    'NYY': 1.18,  # Yankee Stadium — short porch in RF
    'PHI': 1.15,  # Citizens Bank Park
    'BAL': 1.13,  # Camden Yards (post-2022 wall change reduced from ~1.20)
    'MIL': 1.10,  # American Family Field
    'CHC': 1.08,  # Wrigley Field — wind dependent
    'TEX': 1.07,  # Globe Life Field
    'HOU': 1.07,  # Minute Maid Park
    'BOS': 1.05,  # Fenway Park (Green Monster eats some HRs)
    
    # Roughly neutral
    'CWS': 1.04,
    'TOR': 1.03,
    'MIN': 1.02,
    'ATL': 1.02,
    'WSH': 1.01,
    'STL': 1.00,
    'KC':  1.00,
    'LAA': 0.99,
    'NYM': 0.99,
    'ARI': 0.98,
    'DET': 0.98,
    'CLE': 0.97,
    'LAD': 0.97,
    
    # Pitcher-friendly
    'TB':  0.95,  # Tropicana Field
    'SD':  0.94,  # Petco Park
    'PIT': 0.93,  # PNC Park
    'MIA': 0.92,  # loanDepot park
    'OAK': 0.91,  # Oakland Coliseum
    'ATH': 0.91,  # alt code for Oakland
    'SEA': 0.89,  # T-Mobile Park
    'SF':  0.85,  # Oracle Park — extreme pitcher park
}


def get_park_hr_factor(team_abbr):
    """Return HR park factor for the team that plays its home games there.
    
    The `team_abbr` should be the HOME team (i.e. the team whose stadium
    is being used for the game). Returns 1.00 for unknown teams.
    """
    if team_abbr is None:
        return 1.0
    return PARK_HR_FACTORS.get(team_abbr.upper(), 1.0)
