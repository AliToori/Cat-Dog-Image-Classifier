from RPS_game import play, quincy, abbey, kris, mrugesh
from RPS import player
from test_module import test

# Test your player against each bot
# Set verbose=True to see the results of each game
play(player, quincy, 1000, verbose=True)
play(player, abbey, 1000, verbose=True)
play(player, kris, 1000, verbose=True)
play(player, mrugesh, 1000, verbose=True)

# Uncomment the following line to run the unit tests
test()