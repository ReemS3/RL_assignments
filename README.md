# Review other groups' submission:
- Review the first assigment of group 37 [click here](https://docs.google.com/document/d/1zavPsYhRTxLldvB2dgnKQT_ywEHMgTMvbyESneLcljM/edit?usp=sharing).

## Review of group 11:

- For tasks 2 and 3 I don't have anything to add or criticise, I believe our groups have mostly the same points written, but I like you're detailed answer and examples in the latter task. For the first task just a few remarks:
  - The assumption of a deterministic opponent is of course a rather strong one, this would mean that our problem rather is "play chess against this very specific and boring opponent" instead of "play chess against others".
  - There might be some problems with your reward dynamics: If an agent could do a check mate in one or three moves (or after capturing a pawn) it should just prefer the later check mate, which is not what a human would define as "best". Also there is an incentive to rather loose in ten moves than to do a remis (chess term for draw) next move. My suggestions would rather be to remove "artificial" rewards for capturing pieces or getting a check (which is most times not a real problem) but to include a negative reward for losing and small positive reward for draw.
