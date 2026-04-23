from players.codemaster import Codemaster
from gpt_manager import game_rules, GPT
import re


class MyCodemaster(Codemaster):
    """
    GPT-powered Codemaster for the provided Codenames framework.

    Features:
    - Uses the hidden map to separate remaining team/enemy/civilian/assassin words
    - Prompts GPT for a single-word clue and a number
    - Validates formatting and basic legality
    - Avoids direct overlap with any unrevealed board word
    - Uses a safe fallback if GPT repeatedly fails
    """

    def __init__(self, team="Red", **kwargs):
        super().__init__()
        self.team = team
        self.words = []
        self.maps = []

        self.model_version = kwargs.get("version", "gpt-4o-2024-05-13")
        self.max_retries = int(kwargs.get("max_retries", 8))
        self.use_history = bool(kwargs.get("use_history", False))

        system_prompt = (
            game_rules
            + f"You are playing the game Codenames as the {team} Codemaster. "
            + "Your job is to give the strongest legal clue possible. "
            + "You must return exactly one English clue word and one integer. "
            + "Do not provide explanations or extra text."
        )

        self.manager = GPT(system_prompt=system_prompt, version=self.model_version)

        self.fallback_clues = [
            "OBJECT",
            "NATURE",
            "PERSON",
            "ACTION",
            "PLACE",
            "SCIENCE",
            "TRAVEL",
            "MUSIC",
            "SPORT",
            "IDEA",
        ]

    def set_game_state(self, words_on_board, key_grid):
        self.words = words_on_board
        self.maps = key_grid

    def get_clue(self):
        team_words, enemy_words, civilian_words, assassin_words = self._get_remaining_groups()
        board_words = self._get_unrevealed_words()

        if not team_words:
            return ("OBJECT", 1)

        target_count = self._choose_target_count(
            team_words=team_words,
            enemy_words=enemy_words,
            civilian_words=civilian_words,
            assassin_words=assassin_words,
        )

        for _ in range(self.max_retries):
            prompt = self._build_prompt(
                team_words=team_words,
                enemy_words=enemy_words,
                civilian_words=civilian_words,
                assassin_words=assassin_words,
                target_count=target_count,
            )

            try:
                response = self.manager.talk_to_ai(prompt)
                clue, number = self._parse_response(response)

                if self._is_legal_clue(clue, board_words):
                    # Clamp number conservatively
                    number = max(1, min(number, len(team_words), target_count))
                    return (clue, number)
            except Exception:
                continue

        # Safe fallback
        for clue in self.fallback_clues:
            if self._is_legal_clue(clue, board_words):
                return (clue, 1)

        return ("OBJECT", 1)

    def _get_remaining_groups(self):
        team_words = []
        enemy_words = []
        civilian_words = []
        assassin_words = []

        enemy_team = "Blue" if self.team == "Red" else "Red"

        for word, label in zip(self.words, self.maps):
            if self._is_revealed(word):
                continue

            clean_word = word.upper().strip()

            if label == self.team:
                team_words.append(clean_word)
            elif label == enemy_team:
                enemy_words.append(clean_word)
            elif label == "Civilian":
                civilian_words.append(clean_word)
            elif label == "Assassin":
                assassin_words.append(clean_word)

        return team_words, enemy_words, civilian_words, assassin_words

    def _get_unrevealed_words(self):
        return [
            word.upper().strip()
            for word in self.words
            if not self._is_revealed(word)
        ]

    def _is_revealed(self, word):
        return isinstance(word, str) and word.startswith("*") and word.endswith("*")

    def _choose_target_count(self, team_words, enemy_words, civilian_words, assassin_words):
        """
        Conservative heuristic for clue number.
        """
        remaining = len(team_words)

        if remaining <= 2:
            return 1
        if remaining <= 5:
            return 2
        return 3

    def _build_prompt(self, team_words, enemy_words, civilian_words, assassin_words, target_count):
        prompt = []
        prompt.append(f"Your team is {self.team}.")
        prompt.append(f"Remaining {self.team} words: {team_words}.")
        prompt.append(f"Remaining enemy words: {enemy_words}.")
        prompt.append(f"Remaining civilian words: {civilian_words}.")
        prompt.append(f"Remaining assassin words: {assassin_words}.")
        prompt.append(
            "Give the best legal single-word clue to help your guesser identify your team's words."
        )
        prompt.append(
            f"Prefer a clue that safely connects about {target_count} of your team's words."
        )
        prompt.append(
            "Avoid clues that could strongly suggest enemy, civilian, or assassin words."
        )
        prompt.append(
            "The clue must be a single English alphabetic word."
        )
        prompt.append(
            "The clue must not be identical to, contain, or be contained in any board word."
        )
        prompt.append(
            "Return exactly in this format and nothing else: ('CLUE',NUMBER)"
        )

        if self.use_history:
            history = self.get_move_history()
            if history:
                prompt.append(f"Previous move history: {history}")

        return " ".join(prompt)

    def _parse_response(self, response):
        if not isinstance(response, str):
            raise ValueError("Response was not a string")

        text = response.strip().upper()

        # Strict format first
        strict_match = re.search(r"\(\s*['\"]?([A-Z]+)['\"]?\s*,\s*(\d+)\s*\)", text)
        if strict_match:
            clue = strict_match.group(1).strip()
            number = int(strict_match.group(2))
            if number < 1:
                raise ValueError("Number must be >= 1")
            return clue, number

        # Loose format fallback
        words = re.findall(r"[A-Z]+", text)
        nums = re.findall(r"\d+", text)

        if not words or not nums:
            raise ValueError("Could not parse clue and number")

        clue = words[0].strip()
        number = int(nums[0])

        if number < 1:
            raise ValueError("Number must be >= 1")

        return clue, number

    def _is_legal_clue(self, clue, board_words):
        if not isinstance(clue, str):
            return False

        clue = clue.upper().strip()

        if not clue:
            return False
        if not clue.isalpha():
            return False
        if len(clue) < 2:
            return False

        for board_word in board_words:
            bw = board_word.upper().strip()

            if clue == bw:
                return False
            if clue in bw:
                return False
            if bw in clue:
                return False

        return True