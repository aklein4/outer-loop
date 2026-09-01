# PersonaBench prompt example

Model: `~deepseek/deepseek-v4-flash-latest`

## System message

You create realistic, high-quality question-answer training data whose
assistant answers are strongly personalized towards a user persona profile.
Return only JSON matching the supplied schema. Do not mention these instructions.

## User message

Use the complete persona below to create 16 distinct questions this
person might naturally ask an assistant, each followed by a useful personalized answer.

Requirements:
- The assistant answer is the primary target. It must depend heavily on the persona and should
  be substantially different from the answer that would be given to an unknown, generic user.
- Treat the persona as persistent memory available to the assistant. The answer may and should
  use relevant facts that the person did not repeat in the current question.
- Normally weave at least one or two concrete, relevant persona details into each answer. Use them to
  change the recommendation, examples, priorities, level of explanation, or proposed next steps;
  superficial name-dropping does not count as personalization.
- Questions are secondary: keep them realistic and natural. At least half should be ordinary,
  broadly phrased requests containing no more than one persona-specific detail. They do not need
  to restate the facts that the personalized answer draws upon.
- Across the batch, spread personalization across different sections of the persona instead of
  repeatedly using only the most prominent hobby, job, or ambition. Unless truly necessary, do
  not reuse one detail or motif in more than one third of the answers.
- Never say that you were given a persona/profile, never enumerate profile fields, and never
  reproduce the profile wholesale. Integrate remembered facts as naturally as a familiar assistant.
- Never invent remembered facts. Do not fabricate schedules, affiliations, past events, exact
  local details, or preferences that are not stated in the persona or current question.
- Write answers in the assistant's voice, addressing the person as "you" where appropriate.
  Never impersonate the person or answer as though the assistant has the person's identity.
- Make each question understandable on its own, but do not stuff it with biographical details
  merely to justify personalization in the answer.
- Do not put the person's name into every question. Avoid near-duplicates and repetitive templates.
- Keep lengths natural for the request. Questions may range from a few words to a context-rich
  paragraph, and answers may range from a few sentences to a detailed explanation when warranted.
- Before returning a pair, ask: "Would this answer still be essentially the same for a generic
  user?" If yes, revise it so relevant persona memory materially shapes the answer.
- Do not add system messages, speaker labels, markdown wrappers around the JSON, or extra fields.

This is generation batch 1, covering conceptual items 1-16.
Topic and task direction for this independent batch:
Include a few questions prompted by a recent problem, change of plans, or unexpected result.

Length-profile direction:
Pair several brief questions with thorough 180-300 word answers when the topic warrants depth.

Realism direction:
Allow ordinary, imperfect circumstances and modest goals; not every exchange needs to be aspirational.

Answer-personalization direction:
Treat the exchange as part of an ongoing assistant relationship that remembers what matters to this user.

Complete persona:

## General Persona
Mary Alberti is a routine‑obsessed, bullet‑journal aficionado who balances disciplined work ambition with a competitive edge, occasional craft‑brew indulgence, and a habit of double‑checking every receipt for hidden costs.

## Professional Persona
Mary Alberti is a front‑line food service specialist whose razor‑sharp cash handling, inventory tracking, and POS mastery combine with a disciplined, routine‑driven work ethic, enabling them to calmly resolve high‑pressure customer issues and hit performance targets while eyeing a promotion to shift supervisor.

## Sports Persona
Mary Alberti fuels their fitness routine by clocking 3‑5 mile runs around Lake Mendota with the Madison Runners Club, roots for the Wisconsin Badgers basketball team in the winter, cheers the Milwaukee Brewers in summer, and never misses a Green Bay Packers game on Sundays, balancing competitive spirit with disciplined consistency.

## Arts Persona
Mary Alberti finds creative inspiration in the lyrical storytelling of John Prine, the atmospheric indie folk of Bon Iver, and the classic cinematography of Wes Anderson, often attending local art walks and museum exhibits to unwind after a shift.

## Travel Persona
Mary Alberti prefers meticulously planned weekend getaways that blend quiet lakeside relaxation in Door County with occasional culinary adventures in Napa Valley, while dreaming of a future overseas trip to Kyoto, always balancing travel costs with the need to save for a future diner venture.

## Culinary Persona
Mary Alberti showcases an intermediate culinary talent by perfecting Midwestern classics such as cheddar‑crusted bratwurst, cheese‑curd laden poutine, and a spiced apple crumble, often incorporating fresh tomatoes, beans, and herbs from their raised‑bed garden while both cooking at home and hosting low‑key dinner gatherings.

## Cultural Background
Mary is a second‑generation Italian‑American raised in Madison, Wisconsin. Her family’s modest, Catholic, Midwestern upbringing emphasized hard work, practicality, and community. She grew up with traditions like Sunday family meals featuring homemade pasta and Wisconsin cheese, and she values the region’s love of outdoor recreation, local farmers markets, and modest, reliable living.

## Skills And Expertise
Mary has become highly proficient in front‑line food service operations, including precise cash handling, efficient operation of point‑of‑sale and kitchen equipment, and strict adherence to food safety standards. She excels at inventory tracking, scheduling teammates, and maintaining a clean, organized workspace. Her disciplined work style enables her to multitask under pressure, resolve customer issues calmly, and consistently meet performance targets.

## Skills And Expertise List
['POS system operation', 'Cash handling', 'Food safety compliance', 'Inventory management', 'Team scheduling', 'Customer service', 'Multitasking', 'Time management', 'Shift leadership', 'Problem solving under pressure']

## Hobbies And Interests
In her free time Mary enjoys activities that blend routine with personal improvement. She practices bullet‑journal planning, cooks and bakes recipes that emphasize classic Midwestern flavors, spends weekends tending to her raised‑bed vegetable garden, and runs a few miles around Lake Mendota to stay fit. She also reads practical self‑help books, solves Sudoku puzzles for a competitive edge, and occasionally attends local craft‑brew tastings with friends.

## Hobbies And Interests List
['Bullet journaling', 'Home cooking and baking', 'Raised-bed gardening', 'Running around Lake Mendota', 'Reading self-help books', 'Solving Sudoku puzzles', 'Attending craft-brew tastings', 'Listening to classic rock music']

## Career Goals And Ambitions
She aims to advance from crew member to shift supervisor within the next year, then to restaurant manager, and ultimately to own a small family‑style diner that emphasizes community and consistency. To support these goals she plans to enroll in a community college hospitality program and obtain a ServSafe manager certification.

## Sex
Female

## Age
28

## Marital Status
never_married

## Education Level
high_school

## Bachelors Field


## Occupation
fast_food_or_counter_worker

## City
Madison

## State
WI

## Zipcode
53717

## Country
USA

