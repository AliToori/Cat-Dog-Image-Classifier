import unittest
from RPS_game import play, quincy, abbey, kris, mrugesh
from RPS import player

class UnitTests(unittest.TestCase):
    def test_player_vs_quincy(self):
        print("Testing player vs Quincy...")
        wins, losses, ties = play(player, quincy, 1000)
        win_rate = wins / (wins + losses + ties)
        self.assertTrue(win_rate >= 0.6, f"Expected win rate of at least 0.6 against Quincy, but got {win_rate}")

    def test_player_vs_abbey(self):
        print("Testing player vs Abbey...")
        wins, losses, ties = play(player, abbey, 1000)
        win_rate = wins / (wins + losses + ties)
        self.assertTrue(win_rate >= 0.6, f"Expected win rate of at least 0.6 against Abbey, but got {win_rate}")

    def test_player_vs_kris(self):
        print("Testing player vs Kris...")
        wins, losses, ties = play(player, kris, 1000)
        win_rate = wins / (wins + losses + ties)
        self.assertTrue(win_rate >= 0.6, f"Expected win rate of at least 0.6 against Kris, but got {win_rate}")

    def test_player_vs_mrugesh(self):
        print("Testing player vs Mrugesh...")
        wins, losses, ties = play(player, mrugesh, 1000)
        win_rate = wins / (wins + losses + ties)
        self.assertTrue(win_rate >= 0.6, f"Expected win rate of at least 0.6 against Mrugesh, but got {win_rate}")

def test():
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    unittest.TextTestRunner().run(suite)

if __name__ == "__main__":
    test()