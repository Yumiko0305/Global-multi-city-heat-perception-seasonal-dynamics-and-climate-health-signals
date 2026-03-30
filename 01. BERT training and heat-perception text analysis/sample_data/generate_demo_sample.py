#!/usr/bin/env python3
"""
generate_demo_sample.py

Generates a reproducible synthetic demonstration dataset for the heat-perception
pipeline.  The dataset is designed so that reviewers can run all seven pipeline
steps (01–07) end-to-end without access to the real Twitter corpus.

PURPOSE
-------
The original training data was collected from Twitter/X and cannot be redistributed
under the Twitter Developer Agreement and Policy (Section II.C). This script creates
a synthetic demonstration dataset that:

  1. Reproduces the label distribution of the real corpus (~38 % positive)
  2. Covers every annotation category in Supplementary Notes S4.2–S4.3
  3. Embeds realistic seasonal signal: northern/southern hemisphere summer months
     carry a higher positive rate, as observed in the actual data
  4. Uses the study's real city reference pool (cities_reference.csv), spanning
     36 countries and 6 continents
  5. Contains 620 fully unique sentences — no sentence is repeated — so that
     Step 03 deduplication does not reduce the corpus size
  6. Is explicitly flagged as synthetic (is_synthetic = 1) and must NOT be used
     for scientific inference

WHAT TO EXPECT WHEN RUNNING THE PIPELINE ON THIS DATA
------------------------------------------------------
The BERT model trained on this 620-example synthetic corpus will NOT reproduce the
performance figures reported in the manuscript (those required 74,938 labelled tweets).
The purpose here is to validate that every pipeline step executes without errors and
produces correctly structured outputs.  For reference performance figures, see
sample_outputs/monthly_test_metrics_reference.csv.

Usage:
    python sample_data/generate_demo_sample.py
    python sample_data/generate_demo_sample.py --output_dir sample_data --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Sentence bank (label, sentence, category)
# All text is original synthetic prose — NOT from the real Twitter corpus.
# Total: 620 unique sentences (no sentence appears more than once).
# ---------------------------------------------------------------------------

POSITIVE_PHYSIOLOGICAL: List[Tuple[int, str, str]] = [
    (1, "I am absolutely drenched in sweat just walking to the corner store today.", "positive_physiological"),
    (1, "Feeling seriously dehydrated after being outside for only 20 minutes.", "positive_physiological"),
    (1, "Started feeling dizzy and had to sit down — this heat is no joke.", "positive_physiological"),
    (1, "My skin is red and my head is pounding from being out in the sun all afternoon.", "positive_physiological"),
    (1, "Mild heatstroke symptoms kicked in after the outdoor market. Had to rest.", "positive_physiological"),
    (1, "Overheated so quickly this morning. Shirt completely soaked by 9am.", "positive_physiological"),
    (1, "Feeling lightheaded and my hands are cramping — need to cool down ASAP.", "positive_physiological"),
    (1, "Throat is so dry and my lips are cracking. Need water every 10 minutes out here.", "positive_physiological"),
    (1, "Heat exhaustion hit me hard today. Sat in the shade for an hour before I felt okay.", "positive_physiological"),
    (1, "Sweaty and clammy all day, can barely function in this humidity.", "positive_physiological"),
    (1, "Woke up with heat rash from sleeping without AC. So uncomfortable.", "positive_physiological"),
    (1, "Felt nauseous the whole commute home because of the heat on the platform.", "positive_physiological"),
    (1, "My heart is racing just from standing outside. It is way too hot today.", "positive_physiological"),
    (1, "Struggling to breathe out here. This heat is suffocating.", "positive_physiological"),
    (1, "Leg cramps from the heat again. Have to keep drinking electrolytes.", "positive_physiological"),
    (1, "Almost fainted at the bus stop. Someone gave me water. Heat is brutal today.", "positive_physiological"),
    (1, "Clammy and overheated after the walk from the car park. Summer is deadly this year.", "positive_physiological"),
    (1, "Heat stress is real. My productivity tanks every time I step outside.", "positive_physiological"),
    (1, "Got heat fatigue from gardening for just 30 minutes. Had to go in.", "positive_physiological"),
    (1, "Skin flushed and feeling the oppressive heat the moment I left the building.", "positive_physiological"),
    (1, "I sweat through my shirt before even reaching the train station.", "positive_physiological"),
    (1, "Hands are shaking and I feel dizzy — I think I overheated on the walk back.", "positive_physiological"),
    (1, "My whole body feels sticky and heavy in this humidity. Cannot stop sweating.", "positive_physiological"),
    (1, "Headache and nausea hit me as soon as I stepped out of the shade today.", "positive_physiological"),
    (1, "Rapid heartbeat and profuse sweating — definitely heat stress. Going indoors now.", "positive_physiological"),
    (1, "Heat rash on my arms again. Third time this summer. This weather is brutal.", "positive_physiological"),
    (1, "I could barely walk to the shop. Felt parched and my vision went blurry.", "positive_physiological"),
    (1, "Muscle cramps in my calves from being out in the heat all morning.", "positive_physiological"),
    (1, "Completely overheated after the school run. Had to lie down when I got home.", "positive_physiological"),
    (1, "Dripping sweat after 5 minutes outside. This heatwave is no joke at all.", "positive_physiological"),
    # --- 30 additional unique physiological sentences ---
    (1, "My face is beet red and I can feel my pulse in my temples from this heat.", "positive_physiological"),
    (1, "Prickly heat on my back all week from the humid weather. Unbearable.", "positive_physiological"),
    (1, "I drank three litres of water today and still feel parched from the heat.", "positive_physiological"),
    (1, "Staggered into a shop to cool down. My vision was going dark from the heat.", "positive_physiological"),
    (1, "Woke up in the night drenched in sweat again. This heat does not let up.", "positive_physiological"),
    (1, "My breathing felt laboured the whole time I was waiting outside in this heat.", "positive_physiological"),
    (1, "Overheated on the walk home. Had to sit on a bench and pour water on my head.", "positive_physiological"),
    (1, "Arms covered in sweat rash from the humidity. It is incredibly uncomfortable.", "positive_physiological"),
    (1, "Felt faint mid-shift from the heat today. Had to take a 30 minute break inside.", "positive_physiological"),
    (1, "My clothes are completely soaked. Just walked 10 minutes in this scorching heat.", "positive_physiological"),
    (1, "Heat exhaustion again — second time this month. Body just cannot take this summer.", "positive_physiological"),
    (1, "Constant headache for three days now from the extreme heat and dehydration.", "positive_physiological"),
    (1, "Fingers swollen from the heat. Had trouble typing this afternoon.", "positive_physiological"),
    (1, "Felt so nauseous waiting for the bus. This midday heat is genuinely dangerous.", "positive_physiological"),
    (1, "My electrolytes are depleted. Cramping badly after the outdoor morning session.", "positive_physiological"),
    (1, "Dizzy and weak from standing in the sun at the market for two hours.", "positive_physiological"),
    (1, "Throat so parched I could barely speak after the walk from the station.", "positive_physiological"),
    (1, "Profusely sweating even while sitting still inside. The humidity is relentless.", "positive_physiological"),
    (1, "Had to call in sick today. Heat stroke symptoms from yesterday have not passed.", "positive_physiological"),
    (1, "Eyelids heavy and body aching — classic signs of heat fatigue after the outdoor event.", "positive_physiological"),
    (1, "Collapsed briefly on the steps from the heat. Thankfully someone helped me.", "positive_physiological"),
    (1, "Pounding headache all morning from sleeping in a room with no ventilation.", "positive_physiological"),
    (1, "Swollen feet and ankles from standing in the heat all day at the outdoor stall.", "positive_physiological"),
    (1, "Felt my heart fluttering from the heat while jogging. Stopped immediately.", "positive_physiological"),
    (1, "Drenched in sweat within minutes of leaving the building. Skin completely clammy.", "positive_physiological"),
    (1, "I had to stop walking and drink water immediately — feeling of overheating was real.", "positive_physiological"),
    (1, "Heat cramps hit my stomach during the afternoon outdoor training session.", "positive_physiological"),
    (1, "Skin burning and stinging from the direct sun exposure today. Pure heat stress.", "positive_physiological"),
    (1, "Lips cracked and tongue dry after just one hour of outdoor work in this heatwave.", "positive_physiological"),
    (1, "Could not stop shivering despite the heat — classic heat exhaustion warning sign.", "positive_physiological"),
]

POSITIVE_PSYCHOLOGICAL: List[Tuple[int, str, str]] = [
    (1, "Cannot concentrate at all. Too hot to think straight today.", "positive_psychological"),
    (1, "Feeling so irritable and short-tempered. The heat is getting to me.", "positive_psychological"),
    (1, "Mentally drained from the heat. Nothing productive is happening today.", "positive_psychological"),
    (1, "This heat is making me anxious. I feel restless and I cannot settle.", "positive_psychological"),
    (1, "Totally overwhelmed by this weather. I need it to cool down or I will lose my mind.", "positive_psychological"),
    (1, "Cannot sleep again. Too hot to lie still even with the fan on all night.", "positive_psychological"),
    (1, "Heat-induced brain fog is real. I keep forgetting what I was doing.", "positive_psychological"),
    (1, "Feeling miserable in this heat. Everything takes twice as long.", "positive_psychological"),
    (1, "Sluggish and fatigued all day because of the temperatures outside.", "positive_psychological"),
    (1, "Too hot to sleep and too tired to function. This heatwave is brutal.", "positive_psychological"),
    (1, "My mood is terrible. Heat always makes me stressed and distracted.", "positive_psychological"),
    (1, "Cannot stop feeling frustrated. The heat is relentless and I am exhausted.", "positive_psychological"),
    (1, "Restless night because of the heat. Cannot think clearly today at all.", "positive_psychological"),
    (1, "Heat-induced anxiety kicked in during my run. Had to stop and go home.", "positive_psychological"),
    (1, "Feeling mentally exhausted from just existing in this temperature.", "positive_psychological"),
    (1, "The heat is making me short-tempered with everyone around me. I hate this weather.", "positive_psychological"),
    (1, "Too hot to focus on anything. I have been staring at this screen for an hour.", "positive_psychological"),
    (1, "Three nights of broken sleep because of the heat. I am completely exhausted.", "positive_psychological"),
    (1, "I feel completely drained today. The heat is sapping every bit of energy I have.", "positive_psychological"),
    (1, "Can barely keep my eyes open. The heat combined with no sleep is destroying me.", "positive_psychological"),
    # --- 30 additional unique psychological sentences ---
    (1, "My patience has completely run out. The unrelenting heat is making me snappy.", "positive_psychological"),
    (1, "Woke up in a terrible mood. Another hot sleepless night is taking its toll.", "positive_psychological"),
    (1, "Brain is completely fried from the heat. I cannot string two thoughts together.", "positive_psychological"),
    (1, "The heat is making me deeply uncomfortable and anxious about going outside.", "positive_psychological"),
    (1, "I snapped at my colleagues again today. I know it is the heat but I feel awful.", "positive_psychological"),
    (1, "Mentally checked out by noon because of the temperature. Useless afternoon.", "positive_psychological"),
    (1, "I am so irritable from the lack of sleep in this heat. Everything annoys me.", "positive_psychological"),
    (1, "Zoning out constantly today. Heat always turns my brain to mush.", "positive_psychological"),
    (1, "Sixth consecutive night of tossing and turning because of this heat. I am done.", "positive_psychological"),
    (1, "Cannot focus on my work at all. My mind keeps drifting from the discomfort.", "positive_psychological"),
    (1, "Feeling claustrophobic and anxious being stuck indoors to avoid the heat outside.", "positive_psychological"),
    (1, "The heat is grinding me down emotionally. I feel low and completely unmotivated.", "positive_psychological"),
    (1, "Distracted the whole afternoon. Heat makes it impossible to concentrate on anything.", "positive_psychological"),
    (1, "My temper is on a hair trigger this week. All because of this relentless heat.", "positive_psychological"),
    (1, "I feel like I am moving through treacle. Heat fatigue has completely set in.", "positive_psychological"),
    (1, "Cannot enjoy anything outdoors anymore. The heat has made me dread going outside.", "positive_psychological"),
    (1, "Heat anxiety is a real thing. I get panicky just thinking about my commute.", "positive_psychological"),
    (1, "Three weeks of this heat and my mental health is genuinely suffering.", "positive_psychological"),
    (1, "My concentration is shot. Every time I go near a window the heat reminds me.", "positive_psychological"),
    (1, "Miserable and unable to sleep. I hate what summer does to my mental state.", "positive_psychological"),
    (1, "Heat always makes me feel hopeless and stuck. Really struggling this week.", "positive_psychological"),
    (1, "I keep drifting off mid-sentence from the heat fatigue. Embarrassing at work.", "positive_psychological"),
    (1, "Feeling completely overwhelmed and restless from another heat-soaked sleepless night.", "positive_psychological"),
    (1, "My mood swings are directly tied to the temperature. Hot days destroy my wellbeing.", "positive_psychological"),
    (1, "I am so fatigued I cried earlier. The heat has completely broken my resilience.", "positive_psychological"),
    (1, "Heat-induced irritability is real and I am struggling to manage it this week.", "positive_psychological"),
    (1, "Everything feels ten times harder in this heat. Cannot function at full capacity.", "positive_psychological"),
    (1, "I have not been this mentally exhausted in years. This heatwave is destroying me.", "positive_psychological"),
    (1, "Laying awake again at 3am, too hot and too uncomfortable to sleep at all.", "positive_psychological"),
    (1, "Short fuse all day because of the heat. Apologised to three people already.", "positive_psychological"),
]

POSITIVE_COPING: List[Tuple[int, str, str]] = [
    (1, "Finally caved and turned on the AC. Too hot to survive otherwise.", "positive_coping"),
    (1, "Seeking shade wherever I can find it. Cannot stand in the sun a second longer.", "positive_coping"),
    (1, "Just took a cold shower to cool down. Second one today already.", "positive_coping"),
    (1, "Stocked up on iced drinks and trying to stay hydrated through this heat.", "positive_coping"),
    (1, "Carrying a portable fan everywhere I go this week. Total life saver.", "positive_coping"),
    (1, "Went to the shopping centre just for the air conditioning honestly.", "positive_coping"),
    (1, "Wearing the lightest clothes I own and still sweating through them.", "positive_coping"),
    (1, "Cooling down with a cold towel on my neck. Works better than expected.", "positive_coping"),
    (1, "Jumped in the pool to escape the heat. Only tolerable option right now.", "positive_coping"),
    (1, "Had to avoid any physical activity today. Too risky in this heat.", "positive_coping"),
    (1, "Applied ice packs to my wrists and neck to try to cool down.", "positive_coping"),
    (1, "We went to find cool spots downtown after the house got unbearable.", "positive_coping"),
    (1, "Misting spray bottle has been my best friend this entire heatwave.", "positive_coping"),
    (1, "Staying indoors and drinking water constantly. Heat is dangerous outside.", "positive_coping"),
    (1, "Air conditioning is on full blast. Cannot cope with this heat any other way.", "positive_coping"),
    (1, "Drinking cold water every 15 minutes just to stay okay out here.", "positive_coping"),
    (1, "Using every fan in the house right now. Still too hot to be comfortable.", "positive_coping"),
    (1, "Skipped the gym. Not risking heat exhaustion in this weather.", "positive_coping"),
    (1, "Moved my entire work setup to the basement to escape the heat upstairs.", "positive_coping"),
    (1, "Eating cold foods all day. Even cooking makes the apartment unbearable.", "positive_coping"),
    (1, "I opened every window but there is no breeze. Had to turn on the AC after all.", "positive_coping"),
    (1, "Bought a cooling vest today just to get through my outdoor commute.", "positive_coping"),
    (1, "We headed to the library for the AC. Working from home is impossible in this heat.", "positive_coping"),
    (1, "Spent the afternoon at the community pool. Only way to survive this heatwave.", "positive_coping"),
    (1, "Had to reschedule my morning jog to 5am just to beat this ridiculous heat.", "positive_coping"),
    (1, "Putting ice cubes in my water bottle every hour. Still feel like I am melting.", "positive_coping"),
    (1, "Wet towel on my neck is the only thing keeping me going in this heat.", "positive_coping"),
    (1, "Took a cold bath after work. First time I have felt human all day.", "positive_coping"),
    (1, "Staying in the shade the whole lunch break. Direct sun is genuinely dangerous today.", "positive_coping"),
    (1, "I am sleeping on the kitchen floor tonight. Only cool spot in the whole flat.", "positive_coping"),
    # --- 30 additional unique coping sentences ---
    (1, "Put frozen gel packs under my pillow just to get through the night in this heat.", "positive_coping"),
    (1, "Ordered delivery instead of cooking. Even the stove is too much heat right now.", "positive_coping"),
    (1, "Draping a damp sheet over myself at night. Only way to sleep in this heat.", "positive_coping"),
    (1, "Rearranged my whole schedule to avoid being outside between 11am and 4pm.", "positive_coping"),
    (1, "Keeping electrolyte sachets in my bag this entire summer because of the heat.", "positive_coping"),
    (1, "Hung blackout curtains to keep the sun out. Room is still stuffy but slightly better.", "positive_coping"),
    (1, "I drink a cold water bottle before leaving the house just to start cool.", "positive_coping"),
    (1, "Splashing cold water on my face every 20 minutes just to function at work.", "positive_coping"),
    (1, "Changed into dry clothes three times today from sweating so much in this heat.", "positive_coping"),
    (1, "We drove around with the car AC on for 30 minutes just to cool down together.", "positive_coping"),
    (1, "Set up a small paddling pool in the garden. Only tolerable outdoor option today.", "positive_coping"),
    (1, "Holding a cold can of sparkling water against my forehead. Whatever it takes.", "positive_coping"),
    (1, "Called in to work remotely to avoid the commute in this dangerous heat.", "positive_coping"),
    (1, "Cancelled all outdoor plans this week. Not safe to be out in this kind of heat.", "positive_coping"),
    (1, "I switched to linen bedding during the heatwave. Small difference but it helps.", "positive_coping"),
    (1, "Brought a frozen water bottle to the office to keep cool at my desk.", "positive_coping"),
    (1, "Took the shaded side of every street today. Route planning just to avoid the sun.", "positive_coping"),
    (1, "Wrapped a cold flannel around my wrists. Old trick but it works in this heat.", "positive_coping"),
    (1, "Finished work early today specifically to get home before the peak heat hours.", "positive_coping"),
    (1, "I have been wearing a hat every single day this summer just to survive outside.", "positive_coping"),
    (1, "Sat next to the air conditioning vent all afternoon to get through the work day.", "positive_coping"),
    (1, "Bought a handheld misting fan this week. Best purchase of the entire summer.", "positive_coping"),
    (1, "Had my third cold shower of the day after getting back from the supermarket run.", "positive_coping"),
    (1, "Keeping the blinds shut all day on the sunny side of the flat to hold the cool air.", "positive_coping"),
    (1, "Wore a cooling neck wrap during the outdoor ceremony. Essential in this heat.", "positive_coping"),
    (1, "Walked inside a supermarket specifically to cool down before continuing my errands.", "positive_coping"),
    (1, "Switched to morning walks only. Midday heat is far too intense to exercise now.", "positive_coping"),
    (1, "Turned off all heat-generating appliances during the hottest part of the day.", "positive_coping"),
    (1, "Put my wrists under cold running water for five minutes. Immediate heat relief.", "positive_coping"),
    (1, "Keeping a spray bottle of water in the fridge to use throughout this heatwave.", "positive_coping"),
]

POSITIVE_AMBIENT_PERSONAL: List[Tuple[int, str, str]] = [
    (1, "It is absolutely scorching outside and I am melting on my way to work.", "positive_ambient_personal"),
    (1, "The heat is so intense today, walking outside feels physically dangerous.", "positive_ambient_personal"),
    (1, "Sweltering out here. I genuinely do not know how people cope without AC.", "positive_ambient_personal"),
    (1, "The temperature hit 42 degrees today and I am struggling to breathe outside.", "positive_ambient_personal"),
    (1, "Record heat this week and I am feeling every degree of it on my commute.", "positive_ambient_personal"),
    (1, "This heatwave is punishing. I have barely left the house in four days.", "positive_ambient_personal"),
    (1, "Standing at the bus stop in this searing heat is genuinely awful.", "positive_ambient_personal"),
    (1, "Hottest day of the year and I had to walk across campus twice. I am done.", "positive_ambient_personal"),
    (1, "The humidity is unbearable today. Everything sticks to me the moment I go out.", "positive_ambient_personal"),
    (1, "Walking home in this blazing heat was a mistake. I am exhausted.", "positive_ambient_personal"),
    (1, "It is boiling out there today. Even the locals say it is unusually hot.", "positive_ambient_personal"),
    (1, "The heat index is off the charts today and I am feeling it fully.", "positive_ambient_personal"),
    (1, "Sizzling heat all week. I have been working from home just to survive.", "positive_ambient_personal"),
    (1, "Torrid conditions out there. My outdoor plans are totally cancelled.", "positive_ambient_personal"),
    (1, "This heatwave is no joke. I can barely make it to the corner without overheating.", "positive_ambient_personal"),
    (1, "Hottest summer on record and I am living through every miserable degree.", "positive_ambient_personal"),
    (1, "Blazing sun with no breeze. I have never felt heat like this in this city before.", "positive_ambient_personal"),
    (1, "Global warming is cooking us alive. I personally feel it every single day.", "positive_ambient_personal"),
    (1, "Temperatures have broken records this month and I cannot handle it anymore.", "positive_ambient_personal"),
    (1, "The heat outside is intense today. Feels like walking into an oven.", "positive_ambient_personal"),
    (1, "I stepped outside for two minutes and felt like I was being roasted alive.", "positive_ambient_personal"),
    (1, "This city has never been this hot in my 15 years here. I am genuinely struggling.", "positive_ambient_personal"),
    (1, "The tarmac was literally shimmering when I walked to the office. Insane heat.", "positive_ambient_personal"),
    (1, "I cannot believe how hot it is right now. I am just sitting here sweating.", "positive_ambient_personal"),
    (1, "Record high today and I had three outdoor meetings. Completely wiped out.", "positive_ambient_personal"),
    (1, "This heatwave has gone on for two weeks now. My body cannot take it anymore.", "positive_ambient_personal"),
    (1, "Summer this year is relentless. I am drenched the second I walk out the door.", "positive_ambient_personal"),
    (1, "The temperature in the shade is still 38 degrees. I cannot cope with this.", "positive_ambient_personal"),
    (1, "Walking across the car park felt like crossing a desert today. Brutal heat.", "positive_ambient_personal"),
    (1, "I work outdoors and today has been the hardest shift of the entire summer.", "positive_ambient_personal"),
    # --- 30 additional unique ambient personal sentences ---
    (1, "The air feels like a furnace today. I can barely stand being outside at all.", "positive_ambient_personal"),
    (1, "It is hotter outside today than it has ever been in my lifetime here.", "positive_ambient_personal"),
    (1, "The heat radiating off the pavement is intense. I can feel it through my shoes.", "positive_ambient_personal"),
    (1, "Even standing in the shade feels dangerous today. The ambient temperature is brutal.", "positive_ambient_personal"),
    (1, "The scorching heat today makes me regret every outdoor commitment I made this week.", "positive_ambient_personal"),
    (1, "I have lived here 20 years and never experienced heat like this. Truly alarming.", "positive_ambient_personal"),
    (1, "The heat dome sitting over us has made even simple tasks feel overwhelming.", "positive_ambient_personal"),
    (1, "Forty degrees in the shade today and I had no choice but to be out in it.", "positive_ambient_personal"),
    (1, "The extreme heat advisory is not a joke. I felt every warning sign on my walk.", "positive_ambient_personal"),
    (1, "This is genuinely the worst heat I have personally experienced in this city.", "positive_ambient_personal"),
    (1, "I had to cancel my plans today. It is physically unsafe to be outside in this.", "positive_ambient_personal"),
    (1, "The high temperature today broke a 50-year record and I was out in all of it.", "positive_ambient_personal"),
    (1, "Scorching from the moment the sun rose. I felt the heat before I even opened the door.", "positive_ambient_personal"),
    (1, "The urban heat island here is savage. It is always a few degrees hotter in my neighbourhood.", "positive_ambient_personal"),
    (1, "I have started dreading mornings because of the heat that is already there at 7am.", "positive_ambient_personal"),
    (1, "Walked outside and immediately felt like I had opened a preheated oven door.", "positive_ambient_personal"),
    (1, "The humidity combined with the heat today made outdoor breathing genuinely difficult.", "positive_ambient_personal"),
    (1, "Three heatwaves already this summer and we are only halfway through the season.", "positive_ambient_personal"),
    (1, "Every day this week has been hotter than the last. I cannot remember a summer like this.", "positive_ambient_personal"),
    (1, "The city is baking. I can feel the heat reflected off every wall and surface.", "positive_ambient_personal"),
    (1, "I had to work outside during the peak heat today. It was genuinely one of the worst days.", "positive_ambient_personal"),
    (1, "The hot air is still today. No breeze at all. It feels suffocating outside.", "positive_ambient_personal"),
    (1, "This is my third city in two years and the heat here is by far the worst I have experienced.", "positive_ambient_personal"),
    (1, "The sun hits my balcony all afternoon and it turns my whole flat into a sauna.", "positive_ambient_personal"),
    (1, "Forty-one degrees and I had to commute by foot. My entire day was ruined by this heat.", "positive_ambient_personal"),
    (1, "I grew up in a hot country but this heatwave here is something else entirely.", "positive_ambient_personal"),
    (1, "The heat dome has been sitting over us for a week now and I am completely worn down.", "positive_ambient_personal"),
    (1, "Stepped outside at dawn and it was already unbearably warm. No relief from this heat.", "positive_ambient_personal"),
    (1, "The outdoor market was like a furnace today. I had to leave after 20 minutes.", "positive_ambient_personal"),
    (1, "Never thought I would say this city is too hot but this summer has completely changed that.", "positive_ambient_personal"),
]

NEGATIVE_POLICY: List[Tuple[int, str, str]] = [
    (0, "Scientists urge world leaders to take immediate action on global warming.", "negative_policy_discourse"),
    (0, "New research confirms climate change is accelerating faster than expected.", "negative_policy_discourse"),
    (0, "The government unveiled a national heat resilience plan for cities.", "negative_policy_discourse"),
    (0, "Urban heat islands are a major focus of the new climate adaptation policy.", "negative_policy_discourse"),
    (0, "Activists gathered to demand climate action at the city hall today.", "negative_policy_discourse"),
    (0, "Heat mitigation strategies are being trialled in several European capitals.", "negative_policy_discourse"),
    (0, "A new report details the economic costs of climate emergency on agriculture.", "negative_policy_discourse"),
    (0, "Carbon emissions targets were discussed at the international summit this week.", "negative_policy_discourse"),
    (0, "Heat warning issued for the southern regions through end of the week.", "negative_policy_discourse"),
    (0, "Weather forecast: record temperatures expected across the northeast corridor.", "negative_policy_discourse"),
    (0, "Officials issued a heat advisory and opened cooling centres across the county.", "negative_policy_discourse"),
    (0, "Breaking: hottest July ever recorded since weather data collection began.", "negative_policy_discourse"),
    (0, "Extreme heat events are becoming more frequent according to climate scientists.", "negative_policy_discourse"),
    (0, "The heat dome over the Pacific Northwest is now a major policy concern.", "negative_policy_discourse"),
    (0, "Temperature record broken again as greenhouse gas levels hit new highs.", "negative_policy_discourse"),
    (0, "Climate advocacy groups call for urgent urban greening to reduce heat.", "negative_policy_discourse"),
    (0, "Weather report: high pressure system will keep temperatures elevated all week.", "negative_policy_discourse"),
    (0, "A new study links urban heat island effect to increased mortality risk.", "negative_policy_discourse"),
    (0, "Heat emergency declaration issued as temperatures are forecast to exceed 45C.", "negative_policy_discourse"),
    (0, "The latest IPCC report highlights heat as a growing global health threat.", "negative_policy_discourse"),
    (0, "Authorities warn residents ahead of next week's forecast heatwave.", "negative_policy_discourse"),
    (0, "City council approves funding for urban tree planting to combat heat.", "negative_policy_discourse"),
    (0, "Energy overload warnings issued as demand spikes during the heatwave.", "negative_policy_discourse"),
    (0, "Power grid management is being tested as air conditioning demand surges.", "negative_policy_discourse"),
    (0, "Heat resilience plans are being implemented across 20 major cities.", "negative_policy_discourse"),
    (0, "New greenhouse gas emission report released by the environmental agency.", "negative_policy_discourse"),
    (0, "Global average temperatures rose by 1.5 degrees above pre-industrial levels.", "negative_policy_discourse"),
    (0, "Heatwave shelters opened across the region as temperatures soar this week.", "negative_policy_discourse"),
    (0, "Mayor pledges heat action plan after the hottest week in city history.", "negative_policy_discourse"),
    (0, "Report: urban heat island effect worsening in 90 percent of major cities.", "negative_policy_discourse"),
    # --- 30 additional unique policy sentences ---
    (0, "The heat wave forecast prompted an emergency session of the city council.", "negative_policy_discourse"),
    (0, "Municipal authorities are rolling out green roofs to address urban heat.", "negative_policy_discourse"),
    (0, "Meteorological service issues red heat alert for three consecutive days.", "negative_policy_discourse"),
    (0, "New climate policy requires all new buildings to install reflective roofing.", "negative_policy_discourse"),
    (0, "Scientists call on policymakers to declare a climate emergency after record highs.", "negative_policy_discourse"),
    (0, "Heat preparedness campaign launched by the ministry of public health today.", "negative_policy_discourse"),
    (0, "Regional government extends heat emergency measures through the end of August.", "negative_policy_discourse"),
    (0, "Research shows urban greenery can reduce local temperatures by up to 3 degrees.", "negative_policy_discourse"),
    (0, "Climate summit participants agreed to strengthen heat adaptation commitments.", "negative_policy_discourse"),
    (0, "National weather service upgrades heat warning system ahead of summer season.", "negative_policy_discourse"),
    (0, "Cross-party support growing for a mandatory cooling standard in rented housing.", "negative_policy_discourse"),
    (0, "City releases annual heatwave preparedness report ahead of the summer season.", "negative_policy_discourse"),
    (0, "Heat health action plan updated to reflect new research on vulnerable populations.", "negative_policy_discourse"),
    (0, "Local government commits to planting 10,000 trees to reduce urban heat islands.", "negative_policy_discourse"),
    (0, "Climate scientist warns that current heatwaves are a preview of the new normal.", "negative_policy_discourse"),
    (0, "International panel agrees on new thresholds for dangerous heat event classification.", "negative_policy_discourse"),
    (0, "Public health officials issue heat safety guidance for outdoor workers.", "negative_policy_discourse"),
    (0, "New legislation requires employers to provide cooling breaks in extreme heat.", "negative_policy_discourse"),
    (0, "Cities share best practices at global urban heat resilience conference.", "negative_policy_discourse"),
    (0, "Heatwave emergency protocol activated as temperatures breach 40C for the fifth day.", "negative_policy_discourse"),
    (0, "Report calls for mandatory heat impact assessments in urban planning decisions.", "negative_policy_discourse"),
    (0, "Climate change models predict heatwave frequency will triple by 2050.", "negative_policy_discourse"),
    (0, "Heat index methodology updated by meteorological bodies to better reflect humidity.", "negative_policy_discourse"),
    (0, "Government to distribute free cooling kits to vulnerable households this summer.", "negative_policy_discourse"),
    (0, "Research grant awarded to study climate-health links in heat vulnerable cities.", "negative_policy_discourse"),
    (0, "Nature-based solutions to urban heat are gaining traction in city planning.", "negative_policy_discourse"),
    (0, "Emergency services placed on heightened alert during the extended heat period.", "negative_policy_discourse"),
    (0, "Heat mortality risk maps released to help cities prioritise cooling infrastructure.", "negative_policy_discourse"),
    (0, "International aid organisations provide heat relief resources to affected regions.", "negative_policy_discourse"),
    (0, "National adaptation strategy lists heat as the number one climate risk for cities.", "negative_policy_discourse"),
]

NEGATIVE_METAPHORICAL: List[Tuple[int, str, str]] = [
    (0, "That song is so hot right now. Everyone is playing it non-stop.", "negative_metaphorical"),
    (0, "Hot take: the new season was better than everyone is giving it credit for.", "negative_metaphorical"),
    (0, "This debate is heating up and getting very interesting.", "negative_metaphorical"),
    (0, "She is absolutely on fire with her performances this year.", "negative_metaphorical"),
    (0, "Hot off the press — the new album dropped last night and it is incredible.", "negative_metaphorical"),
    (0, "This topic is so hot right now in the tech world.", "negative_metaphorical"),
    (0, "Things are heating up between the two candidates ahead of the debate.", "negative_metaphorical"),
    (0, "That was a heated argument at the board meeting yesterday.", "negative_metaphorical"),
    (0, "She looked so hot at the gala last night. Stunning dress.", "negative_metaphorical"),
    (0, "Hot mess energy from that presentation. Completely all over the place.", "negative_metaphorical"),
    (0, "The rivalry between the two teams is really heating up this season.", "negative_metaphorical"),
    (0, "He is in the hot seat after those controversial comments last week.", "negative_metaphorical"),
    (0, "Hot girl summer vibes — everyone is out and living their best life.", "negative_metaphorical"),
    (0, "This stock is smoking hot right now. Investors cannot get enough.", "negative_metaphorical"),
    (0, "The political situation is getting hotter by the day.", "negative_metaphorical"),
    (0, "Under the heat of scrutiny, the CEO was forced to resign.", "negative_metaphorical"),
    (0, "That colour combo is so hot this season. Everywhere on the runways.", "negative_metaphorical"),
    (0, "Hot and trending: this video has already hit 10 million views.", "negative_metaphorical"),
    (0, "The market is red hot right now. Best time to invest in years.", "negative_metaphorical"),
    (0, "Those dance moves were absolutely fire. The crowd went wild.", "negative_metaphorical"),
    (0, "This is the hottest ticket in town. Sold out in 10 minutes.", "negative_metaphorical"),
    (0, "She is seriously on fire this year. Award after award.", "negative_metaphorical"),
    (0, "Hot competition between the two brands is good for consumers.", "negative_metaphorical"),
    (0, "The conversation is getting heated. Both sides are passionate about this.", "negative_metaphorical"),
    (0, "That track is absolutely blazing. It will be song of the summer easily.", "negative_metaphorical"),
    # --- 35 additional unique metaphorical sentences ---
    (0, "He really turned up the heat in that negotiation. Bold move.", "negative_metaphorical"),
    (0, "White hot anticipation building for the championship final tonight.", "negative_metaphorical"),
    (0, "Their chemistry on screen is absolutely scorching. Perfect casting.", "negative_metaphorical"),
    (0, "The pressure is really heating up on the management team right now.", "negative_metaphorical"),
    (0, "She dropped a hot take on the podcast and everyone is talking about it.", "negative_metaphorical"),
    (0, "Things got pretty heated at the town hall meeting last night.", "negative_metaphorical"),
    (0, "The competition is fierce and the stakes are burning hot this quarter.", "negative_metaphorical"),
    (0, "He walked into that interview blazing with confidence. Nailed it.", "negative_metaphorical"),
    (0, "The atmosphere in the stadium was electric and absolutely red hot.", "negative_metaphorical"),
    (0, "Her fashion sense is on fire this season. Every outfit is stunning.", "negative_metaphorical"),
    (0, "That business idea is hot property right now. Everyone wants a piece.", "negative_metaphorical"),
    (0, "The tension between the two departments has been simmering for months.", "negative_metaphorical"),
    (0, "His career trajectory is blazing right now. Biggest star in the industry.", "negative_metaphorical"),
    (0, "The bidding war for the acquisition target is heating up fast.", "negative_metaphorical"),
    (0, "Sizzling chemistry between the leads makes this drama unmissable viewing.", "negative_metaphorical"),
    (0, "Hot button issue this election cycle is housing affordability.", "negative_metaphorical"),
    (0, "Their rivalry is scorching. Every interaction between them is tense.", "negative_metaphorical"),
    (0, "The debate over the new policy is getting very heated indeed.", "negative_metaphorical"),
    (0, "She is running hot this season. Three wins in a row already.", "negative_metaphorical"),
    (0, "Hot tip from a reliable source: the merger announcement comes this Friday.", "negative_metaphorical"),
    (0, "The tension in the room was palpable — a truly heated exchange.", "negative_metaphorical"),
    (0, "He is in a hot streak and looks unstoppable heading into the finals.", "negative_metaphorical"),
    (0, "That performance was pure fire. The audience were on their feet.", "negative_metaphorical"),
    (0, "Things are really coming to a boil in that organisation. Big changes ahead.", "negative_metaphorical"),
    (0, "Hot topic of the week: should remote work remain the norm permanently?", "negative_metaphorical"),
    (0, "The election race is heating up dramatically as polling day approaches.", "negative_metaphorical"),
    (0, "Her social media presence is absolutely on fire this month.", "negative_metaphorical"),
    (0, "A blazing row erupted in parliament over the proposed budget cuts.", "negative_metaphorical"),
    (0, "The new restaurant is the hottest reservation in the city right now.", "negative_metaphorical"),
    (0, "Under pressure and in the heat of the moment he made the wrong call.", "negative_metaphorical"),
    (0, "The board really turned up the heat on the CEO in that quarterly review.", "negative_metaphorical"),
    (0, "She delivered that speech with white hot intensity. Truly powerful.", "negative_metaphorical"),
    (0, "This project is sizzling with potential. I cannot wait to see how it develops.", "negative_metaphorical"),
    (0, "The critics are burning hot on this release. Every review is five stars.", "negative_metaphorical"),
    (0, "Things have really heated up between them since the merger was announced.", "negative_metaphorical"),
]

NEGATIVE_PRODUCT_MEDIA: List[Tuple[int, str, str]] = [
    (0, "The movie Heat is one of Al Pacino's finest performances ever.", "negative_media_product"),
    (0, "I finally watched Heat last night. The diner scene is iconic.", "negative_media_product"),
    (0, "The album drops next Friday. The lead single is called Summer Heat.", "negative_media_product"),
    (0, "New episode of Heatwave the series is out. Do not spoil it for me.", "negative_media_product"),
    (0, "Just finished the Hot Stuff podcast episode — really well produced.", "negative_media_product"),
    (0, "Huge sale on hot deals this weekend. Up to 70 percent off storewide.", "negative_commercial_business"),
    (0, "Hot market conditions mean prices are rising in every major city.", "negative_commercial_business"),
    (0, "Selling like hotcakes — the product sold out within 30 minutes of launch.", "negative_commercial_business"),
    (0, "That is a hot property in this neighbourhood. Will not last long.", "negative_commercial_business"),
    (0, "Hot stock picks for Q3 — analysts are very bullish on these.", "negative_commercial_business"),
    (0, "Limited hot offers this week only. Click the link in bio for details.", "negative_commercial_business"),
    (0, "Hot leads from the conference — three solid prospects already.", "negative_commercial_business"),
    (0, "The GPU is overheating again under load. Need to reapply the thermal paste.", "negative_technical_scientific"),
    (0, "Heat transfer efficiency improved by 18 percent with the new alloy design.", "negative_technical_scientific"),
    (0, "Specific heat capacity calculation for the new material is very promising.", "negative_technical_scientific"),
    (0, "We installed a new heat exchanger in the industrial cooling unit today.", "negative_technical_scientific"),
    (0, "Hot swapping the drives while the server is live saved us three hours.", "negative_technical_scientific"),
    (0, "Latent heat release during condensation is driving the storm development.", "negative_technical_scientific"),
    (0, "The heat pump is far more efficient than the old boiler for winter heating.", "negative_technical_scientific"),
    (0, "Heat capacity of the container walls needs to be factored into the model.", "negative_technical_scientific"),
    (0, "Looking forward to the Hot Ones interview later tonight. Great show.", "negative_media_product"),
    (0, "Just got the Hot Wheels limited edition set. Worth every penny.", "negative_media_product"),
    (0, "Heat the film has the best heist scene in cinema history, no contest.", "negative_media_product"),
    (0, "Fire up the grill — who wants hot dogs at the barbecue tonight?", "negative_food"),
    (0, "Hot dog stand near the stadium has the best onions. Worth the queue.", "negative_food"),
    # --- 35 additional unique product/media/technical sentences ---
    (0, "Just re-watched Heat for the fifth time. Mann is a genius director.", "negative_media_product"),
    (0, "The Hot 100 this week is dominated by one artist. Impressive chart run.", "negative_media_product"),
    (0, "Summer Heat is a great album title. Already streaming it on repeat.", "negative_media_product"),
    (0, "That Heatwave documentary is streaming now. Really eye-opening about music history.", "negative_media_product"),
    (0, "Just bought the new Hot Wheels track set for my nephew. He will love it.", "negative_media_product"),
    (0, "New gaming laptop keeps overheating during extended sessions. Annoying design flaw.", "negative_technical_scientific"),
    (0, "The thermal management on this CPU cooler is impressive at full load.", "negative_technical_scientific"),
    (0, "Our server room heat dissipation system was upgraded to handle the new hardware.", "negative_technical_scientific"),
    (0, "Heat sink dimensions need to be recalculated for the revised chip layout.", "negative_technical_scientific"),
    (0, "The specific heat ratio assumption in the model was off by a factor of two.", "negative_technical_scientific"),
    (0, "Thermodynamic efficiency of the heat recovery system exceeded projections.", "negative_technical_scientific"),
    (0, "The furnace heat distribution was uneven — rebalancing the ductwork fixed it.", "negative_technical_scientific"),
    (0, "GPU temperature spiked to 95C during the benchmark. Thermal throttling kicked in.", "negative_technical_scientific"),
    (0, "Hot reload feature in the new framework saves so much development time.", "negative_technical_scientific"),
    (0, "Flash sale on hot electronics starting midnight. Deals include laptops and tablets.", "negative_commercial_business"),
    (0, "This product is selling hot right now. Pre-orders have exceeded all forecasts.", "negative_commercial_business"),
    (0, "Hot deals newsletter dropped today. Some incredible discounts this week.", "negative_commercial_business"),
    (0, "Our hottest selling item this quarter is the portable speaker. Flying off shelves.", "negative_commercial_business"),
    (0, "Investor interest in this sector is red hot. Valuations are through the roof.", "negative_commercial_business"),
    (0, "Hot new startup in the fintech space just raised a massive Series B round.", "negative_commercial_business"),
    (0, "The commercial real estate market is on fire right now. Best time to sell.", "negative_commercial_business"),
    (0, "That cafe does the best hot sandwiches in the neighbourhood. Worth the wait.", "negative_food"),
    (0, "Made a batch of hot honey at home this weekend. Goes with everything.", "negative_food"),
    (0, "The hot and sour soup at that restaurant is absolutely outstanding.", "negative_food"),
    (0, "Tried the new hot pot place downtown. Amazing broth selection.", "negative_food"),
    (0, "These hot chips with aioli are the ultimate comfort food. Obsessed.", "negative_food"),
    (0, "Fresh hot cinnamon rolls from the bakery this morning. Perfect Sunday treat.", "negative_food"),
    (0, "The hot matcha latte at this place is genuinely the best I have had.", "negative_food"),
    (0, "Homemade hot sauce recipe turned out incredible. Sharing it with everyone.", "negative_food"),
    (0, "The heat treatment process for the new alloy is being optimised in the lab.", "negative_technical_scientific"),
    (0, "Radio heat map shows frequency interference in the northeast quadrant.", "negative_technical_scientific"),
    (0, "Thermal imaging revealed two hot spots in the building's insulation layer.", "negative_technical_scientific"),
    (0, "Hot standby failover worked perfectly during the simulated outage drill.", "negative_technical_scientific"),
    (0, "Heat tracing cables installed along the exterior pipes to prevent freezing.", "negative_technical_scientific"),
    (0, "The new Hot 97 playlist just dropped and it is fire from start to finish.", "negative_media_product"),
]

NEGATIVE_INDOOR_FOOD: List[Tuple[int, str, str]] = [
    (0, "The central heating is finally working again after the engineer fixed it.", "negative_indoor_heating"),
    (0, "Turned down the radiator in the bedroom. It was way too warm in there.", "negative_indoor_heating"),
    (0, "The heater broke at the worst time. Freezing cold all week indoors.", "negative_indoor_heating"),
    (0, "Our home turned into a sauna with the central heating on full blast.", "negative_indoor_heating"),
    (0, "Warm indoors but absolutely freezing outside. Classic winter.", "negative_indoor_heating"),
    (0, "The radiator is making strange noises again. Need to bleed it.", "negative_indoor_heating"),
    (0, "Ordered hot pot for dinner tonight. The broth was perfect.", "negative_food"),
    (0, "Made hot chocolate from scratch today. Rich and creamy.", "negative_food"),
    (0, "These hot wings are too spicy for me but I cannot stop eating them.", "negative_food"),
    (0, "The spicy heat in this curry is unreal. My mouth is on fire.", "negative_food"),
    (0, "Hot cross buns fresh from the oven. Best Easter tradition.", "negative_food"),
    (0, "Tried the hot tamales from the new place. Absolutely delicious.", "negative_food"),
    (0, "Hot cider on a cold evening is the best thing in the world.", "negative_food"),
    (0, "That hot sauce is too much even for me. Three stars maximum.", "negative_food"),
    (0, "Home heating bill went up again this winter. Really feeling the cost.", "negative_indoor_heating"),
    (0, "New radiator installed in the living room. Finally warm indoors again.", "negative_indoor_heating"),
    # --- 34 additional unique indoor/food sentences ---
    (0, "The underfloor heating in the new flat is an absolute luxury.", "negative_indoor_heating"),
    (0, "Our district heating system went down for maintenance this morning.", "negative_indoor_heating"),
    (0, "Scheduled the annual boiler service before the heating season starts.", "negative_indoor_heating"),
    (0, "The smart thermostat for the central heating has been a game changer.", "negative_indoor_heating"),
    (0, "Installed insulation in the attic last month. Heating bills dropped significantly.", "negative_indoor_heating"),
    (0, "The storage heaters in this flat are ancient and inefficient.", "negative_indoor_heating"),
    (0, "Had the heating system flushed and balanced. Much more even warmth now.", "negative_indoor_heating"),
    (0, "The gas heating is still cheaper than electric in this area.", "negative_indoor_heating"),
    (0, "Hot toddy is my go-to cold remedy during winter. Works every time.", "negative_food"),
    (0, "The hot ramen at that place is worth every penny. Incredibly rich broth.", "negative_food"),
    (0, "Made a big batch of hot lentil soup for the week ahead. So satisfying.", "negative_food"),
    (0, "The jalapeño heat in this salsa is perfect. Not too much, not too little.", "negative_food"),
    (0, "Hot stone massage at the spa was the most relaxing thing I have done all year.", "negative_food"),
    (0, "Fresh hot bread from the bakery downstairs every morning. Best part of my day.", "negative_food"),
    (0, "The chilli heat level in that dish was intense. Had to drink a lot of milk.", "negative_food"),
    (0, "Slow-cooked hot pot with my family on Sunday. Perfect winter comfort food.", "negative_food"),
    (0, "My heating engineer recommended a full system flush every five years.", "negative_indoor_heating"),
    (0, "The oil heating tank needs refilling before the cold snap arrives.", "negative_indoor_heating"),
    (0, "New heat recovery ventilation system installed in the extension last week.", "negative_indoor_heating"),
    (0, "The pilot light on the old heating boiler keeps going out in cold weather.", "negative_indoor_heating"),
    (0, "A hot ginger tea is the perfect thing on a cold and rainy afternoon.", "negative_food"),
    (0, "Hot pierogies with sour cream — best comfort food from my grandmother's recipe.", "negative_food"),
    (0, "The kitchen was warming up nicely with the oven on for the Sunday roast.", "negative_indoor_heating"),
    (0, "Heat settings on the oven were off. The roast took an extra 30 minutes.", "negative_indoor_heating"),
    (0, "The fireplace heats the living room better than any radiator could.", "negative_indoor_heating"),
    (0, "Spice heat tolerance builds over time. I can handle much hotter food now.", "negative_food"),
    (0, "Hot pastrami on rye is the only sandwich worth ordering at this deli.", "negative_food"),
    (0, "Homemade hot mustard recipe finally nailed it after three attempts.", "negative_food"),
    (0, "The chimney sweep came today before we start using the wood heating again.", "negative_indoor_heating"),
    (0, "We had hot mulled wine at the Christmas market. Absolutely delicious.", "negative_food"),
    (0, "Infrared heating panels are cheaper to run than traditional radiators.", "negative_indoor_heating"),
    (0, "The communal heating in our block has been unreliable all winter.", "negative_indoor_heating"),
    (0, "Had the best hot breakfast at the café near the station this morning.", "negative_food"),
    (0, "Hot honey drizzled over pizza is my new favourite food combination.", "negative_food"),
]

NEGATIVE_PROPER_NOUN: List[Tuple[int, str, str]] = [
    (0, "Miami Heat had an incredible run this season. Championship contenders.", "negative_proper_noun"),
    (0, "The Miami Heat roster looks strong heading into the playoffs.", "negative_proper_noun"),
    (0, "Going to Hot Springs this weekend for the mineral baths. Cannot wait.", "negative_proper_noun"),
    (0, "The Billboard Hot 100 chart this week is dominated by pop acts.", "negative_proper_noun"),
    (0, "Hot air balloon festival starts this Saturday. Tickets are sold out.", "negative_proper_noun"),
    (0, "Death Valley is on my travel bucket list. The landscape looks surreal.", "negative_proper_noun"),
    (0, "Heat Magazine's latest cover is absolutely stunning. Great photoshoot.", "negative_proper_noun"),
    (0, "Heatwave Festival lineup announced and it looks absolutely incredible.", "negative_proper_noun"),
    (0, "The Miami Heat coach gave a great press conference after the win.", "negative_proper_noun"),
    (0, "Hot Springs National Park trail was beautiful. Highly recommend.", "negative_proper_noun"),
    (0, "Miami Heat vs Lakers tonight. Big game. Cannot wait to watch.", "negative_proper_noun"),
    (0, "Booked a trip to Hot Springs. Heard the baths are incredible.", "negative_proper_noun"),
    # --- 28 additional unique proper noun sentences ---
    (0, "The Oklahoma City Thunder beat the Miami Heat in an overtime thriller.", "negative_proper_noun"),
    (0, "Heatwave Music Festival sold out in under two hours. Insane demand.", "negative_proper_noun"),
    (0, "Just signed up for the Hot Yoga class downtown. Starting next Monday.", "negative_proper_noun"),
    (0, "Heat Street the news outlet published a great analysis piece yesterday.", "negative_proper_noun"),
    (0, "Hot Topic store in the mall has amazing alternative fashion. Love it.", "negative_proper_noun"),
    (0, "The Hot Springs resort is fully booked until April. Plan early.", "negative_proper_noun"),
    (0, "Miami Heat ticket prices have gone through the roof this season.", "negative_proper_noun"),
    (0, "Hot Creek Geological Site in California is worth visiting if you are nearby.", "negative_proper_noun"),
    (0, "The Billboard Hot 100 is dominated by that one track for the third week.", "negative_proper_noun"),
    (0, "Death Valley set another temperature record according to the park service.", "negative_proper_noun"),
    (0, "Yellowstone hot springs area was breath-taking. Truly unique landscape.", "negative_proper_noun"),
    (0, "Heat magazine ran a feature on celebrity wellness this issue.", "negative_proper_noun"),
    (0, "Hot Fuzz is one of the funniest British comedies ever made. Classic.", "negative_proper_noun"),
    (0, "The Hot Springs marathon is one of the most scenic races in the country.", "negative_proper_noun"),
    (0, "Just discovered the Heat podcast. Already through four episodes.", "negative_proper_noun"),
    (0, "Miami Heat preseason starts next week. Cautiously optimistic about this squad.", "negative_proper_noun"),
    (0, "Heatwave Bar downtown has the best cocktails. Packed every weekend.", "negative_proper_noun"),
    (0, "The Hot Lake Springs hotel is a surprisingly luxurious historic gem.", "negative_proper_noun"),
    (0, "Boiling Lake in Dominica is on my must-see natural wonder list.", "negative_proper_noun"),
    (0, "Thermopolis Wyoming hot springs is the world's largest mineral hot spring.", "negative_proper_noun"),
    (0, "Scorching Desert Ultra is the toughest race I have ever attempted.", "negative_proper_noun"),
    (0, "The Desert Heat film festival had an incredible programme this year.", "negative_proper_noun"),
    (0, "Hot Wheels collecting community is incredibly passionate and knowledgeable.", "negative_proper_noun"),
    (0, "Attending the Hot Air Balloon Fiesta this October. Bucket list item ticked.", "negative_proper_noun"),
    (0, "The Heatwave music venue has hosted some legendary nights over the years.", "negative_proper_noun"),
    (0, "Furnace Creek in Death Valley is reportedly one of the hottest spots on Earth.", "negative_proper_noun"),
    (0, "Miami Heat academy is producing some of the best young talent in the league.", "negative_proper_noun"),
    (0, "The Hot 97 Summer Jam lineup this year is absolutely stacked with talent.", "negative_proper_noun"),
]

NEGATIVE_SPAM: List[Tuple[int, str, str]] = [
    (0, "Loving this new song!! #bts #kpop #heatwave #summer #trending", "negative_spam_irrelevant"),
    (0, "Retweet if you had a great weekend! #heatwave #vibes #goodtimes", "negative_spam_irrelevant"),
    (0, "Follow for follow! New account looking to connect. #heatwave #music", "negative_spam_irrelevant"),
    (0, "Check out our new single dropping Friday! #heatwave #newmusic #pop", "negative_spam_irrelevant"),
    (0, "Birthday shoutout to my bestie! Best day ever! #heatwave #celebrations", "negative_spam_irrelevant"),
    (0, "Just posted a new TikTok! Link in bio. #heatwave #viral #trending", "negative_spam_irrelevant"),
    (0, "Gorgeous sunset tonight from my balcony. #heatwave #photography #golden", "negative_spam_irrelevant"),
    (0, "Game night with the crew! #heatwave #boardgames #weekend", "negative_spam_irrelevant"),
    (0, "New gym PR today! Hard work pays off. #heatwave #fitness #gains", "negative_spam_irrelevant"),
    (0, "Monthly giveaway! RT to enter. #heatwave #giveaway #freebie", "negative_spam_irrelevant"),
    (0, "Friday night plans sorted. Who is out this weekend? #heatwave", "negative_spam_irrelevant"),
    (0, "New look dropped today. Check the store. #heatwave #fashion #style", "negative_spam_irrelevant"),
    (0, "Stream our latest EP now on all platforms. #heatwave #indie #newmusic", "negative_spam_irrelevant"),
    (0, "Afternoon vibes. Chilling with the gang. #heatwave #summer #friends", "negative_spam_irrelevant"),
    (0, "Good morning everyone! Make today count. #heatwave #motivation", "negative_spam_irrelevant"),
    # --- 25 additional unique spam sentences ---
    (0, "Just hit 10k followers! Thank you all so much! #heatwave #milestone", "negative_spam_irrelevant"),
    (0, "Only 3 spots left in my online course. DM me! #heatwave #coaching", "negative_spam_irrelevant"),
    (0, "Big announcement coming tomorrow! Stay tuned. #heatwave #exciting", "negative_spam_irrelevant"),
    (0, "Throwback to last summer! Good times. #heatwave #tbt #memories", "negative_spam_irrelevant"),
    (0, "Coffee and music. That is all I need today. #heatwave #chill #morning", "negative_spam_irrelevant"),
    (0, "Just landed! So excited to be here. #heatwave #travel #adventure", "negative_spam_irrelevant"),
    (0, "New blog post is live! Check the link in bio. #heatwave #lifestyle", "negative_spam_irrelevant"),
    (0, "Dog mum life is the best life. #heatwave #dogs #dogsofinstagram", "negative_spam_irrelevant"),
    (0, "Last day to grab early bird tickets! #heatwave #event #dontmissout", "negative_spam_irrelevant"),
    (0, "Grateful for every single day. Positivity only. #heatwave #mindset", "negative_spam_irrelevant"),
    (0, "My bestie just got engaged! So happy for them. #heatwave #love", "negative_spam_irrelevant"),
    (0, "Collaboration alert! Exciting news coming soon. #heatwave #collab", "negative_spam_irrelevant"),
    (0, "This playlist is everything right now. Link in bio. #heatwave #music", "negative_spam_irrelevant"),
    (0, "Dropping new content every day this week. #heatwave #creator #daily", "negative_spam_irrelevant"),
    (0, "Rate my fit! Feeling this look today. #heatwave #ootd #fashion", "negative_spam_irrelevant"),
    (0, "Just finished a great book! Highly recommend it. #heatwave #reading", "negative_spam_irrelevant"),
    (0, "Win a free product bundle! RT and follow to enter. #heatwave #giveaway", "negative_spam_irrelevant"),
    (0, "Life is good when you have great friends. #heatwave #blessed #grateful", "negative_spam_irrelevant"),
    (0, "New week new goals. Let us make it happen! #heatwave #mondaymotivation", "negative_spam_irrelevant"),
    (0, "Recipe drop today! Check the pinned post. #heatwave #food #recipes", "negative_spam_irrelevant"),
    (0, "Sunset views from the rooftop. Cannot get enough. #heatwave #views", "negative_spam_irrelevant"),
    (0, "Just signed with a new agency! Big things coming. #heatwave #exciting", "negative_spam_irrelevant"),
    (0, "Cute café find in the neighbourhood. Obsessed. #heatwave #café #vibes", "negative_spam_irrelevant"),
    (0, "Merch drop this Friday! Limited stock so be quick. #heatwave #merch", "negative_spam_irrelevant"),
    (0, "Podcast episode out now! Link in bio. Listen and subscribe. #heatwave", "negative_spam_irrelevant"),
]

BOUNDARY_POSITIVE: List[Tuple[int, str, str]] = [
    (1, "They issued another heat warning and I am already feeling sick from it.", "boundary_positive_alert_personal"),
    (1, "Record heat this week — I have barely made it through my commute each day.", "boundary_positive_record_personal"),
    (1, "The heatwave is destroying me. I have not slept properly in five nights.", "boundary_positive_heatwave_personal"),
    (1, "Climate change is cooking us — I felt it intensely on my walk this morning.", "boundary_positive_climate_personal_exposure"),
    (1, "Global warming is real and I feel it every time I step outside in summer.", "boundary_positive_climate_first_person"),
    (1, "Volunteers opened cooling centres and I went — I was genuinely struggling.", "boundary_positive_cooling_centre_personal_use"),
    (1, "The heat advisory is in effect and I already have a pounding headache.", "boundary_positive_advisory_symptom"),
    (1, "Temperature record broken today and I am melting out here on the job site.", "boundary_positive_record_outdoor_work"),
    (1, "Even with the heatwave warning I had to work outside. Completely exhausted now.", "boundary_positive_forced_outdoor_exposure"),
    (1, "Everyone is talking about climate change but today I am just trying to survive the walk home.", "boundary_positive_personal_amid_discourse"),
    # --- 30 additional unique boundary positive sentences ---
    (1, "Heat emergency declared and I personally felt every degree of it today.", "boundary_positive_alert_personal"),
    (1, "The heat relief centre opened yesterday and I needed it — I was in a bad way.", "boundary_positive_cooling_centre_personal_use"),
    (1, "Red heat alert today and I was outside for three hours. Not good.", "boundary_positive_alert_personal"),
    (1, "They keep issuing heat advisories and I keep ignoring them. Today I regretted it.", "boundary_positive_advisory_symptom"),
    (1, "Global warming feels very personal when I am sweating through my third shirt.", "boundary_positive_climate_first_person"),
    (1, "The heatwave shelter was a lifesaver today. I walked in barely standing.", "boundary_positive_cooling_centre_personal_use"),
    (1, "Record temperatures all week and my body has just not been able to cope.", "boundary_positive_record_personal"),
    (1, "Climate scientists are warning about this heat and I feel their point exactly.", "boundary_positive_climate_personal_exposure"),
    (1, "The urban green space nearby was my only relief during the heat emergency today.", "boundary_positive_cooling_centre_personal_use"),
    (1, "They warned about extreme heat and they were right. I am completely done in.", "boundary_positive_advisory_symptom"),
    (1, "The hottest day on record and I was stuck outside for my entire shift.", "boundary_positive_record_outdoor_work"),
    (1, "Heat warning levels are abstract until you are the one overheating in the sun.", "boundary_positive_alert_personal"),
    (1, "Climate breakdown hit home for me today when I nearly fainted at the bus stop.", "boundary_positive_climate_personal_exposure"),
    (1, "The forecast said dangerous heat and they were not wrong. I struggled all day.", "boundary_positive_advisory_symptom"),
    (1, "This heat emergency is not a news story for me — it is a daily survival situation.", "boundary_positive_alert_personal"),
    (1, "I visited the cooling centre after the heat got unbearable at home today.", "boundary_positive_cooling_centre_personal_use"),
    (1, "Another heat record and another day of me sweating through everything I own.", "boundary_positive_record_personal"),
    (1, "They opened a heat refuge downtown and I used it. Not ashamed at all.", "boundary_positive_cooling_centre_personal_use"),
    (1, "The heat helpline advised me to rest indoors. I should have called hours earlier.", "boundary_positive_advisory_symptom"),
    (1, "Climate action matters to me — and right now so does surviving the commute in this heat.", "boundary_positive_climate_first_person"),
    (1, "The heat dome is a meteorological event but it felt deeply personal today.", "boundary_positive_alert_personal"),
    (1, "Temperature records are being broken and I am personally suffering every single one.", "boundary_positive_record_personal"),
    (1, "The council opened emergency cooling spots and I went to three of them today.", "boundary_positive_cooling_centre_personal_use"),
    (1, "Heard the heat advisory on the radio and still had to go out. Worst decision.", "boundary_positive_advisory_symptom"),
    (1, "This heatwave is in the news but for me it is just an ordinary Tuesday in agony.", "boundary_positive_heatwave_personal"),
    (1, "Climate crisis feels abstract until you are dizzy and dehydrated on your walk to work.", "boundary_positive_climate_personal_exposure"),
    (1, "The extreme heat bulletin went out at 6am and by 9am I was already struggling.", "boundary_positive_alert_personal"),
    (1, "I went to the community cooling centre for the first time ever today. That bad.", "boundary_positive_cooling_centre_personal_use"),
    (1, "Record highs are just numbers until you have to live and work in them.", "boundary_positive_record_outdoor_work"),
    (1, "The heat action plan says to stay indoors but some of us have no choice.", "boundary_positive_forced_outdoor_exposure"),
]

BOUNDARY_NEGATIVE: List[Tuple[int, str, str]] = [
    (0, "Volunteers helped homeless people stay cool during the downtown heatwave.", "boundary_negative_third_party"),
    (0, "Heat relief centres are now open for vulnerable residents in the city.", "boundary_negative_news_service"),
    (0, "Local authorities urged people to stay indoors during the extreme heat warning.", "boundary_negative_news_advisory"),
    (0, "Elderly residents were transported to cooling shelters during the heatwave.", "boundary_negative_news_third_party"),
    (0, "The community set up fans and water stations for people during the hot spell.", "boundary_negative_community_third_party"),
    (0, "Hospitals reported an increase in heat-related admissions during the weekend.", "boundary_negative_news_health_report"),
    (0, "Schools closed early as temperatures exceeded safe outdoor thresholds.", "boundary_negative_institutional_response"),
    (0, "Heat-related deaths reported as authorities issue extreme weather warning.", "boundary_negative_news_mortality"),
    (0, "City workers were given extra breaks and water during the record heat event.", "boundary_negative_news_welfare_policy"),
    (0, "Emergency services responded to dozens of heat-related calls over the weekend.", "boundary_negative_news_emergency"),
    # --- 30 additional unique boundary negative sentences ---
    (0, "Paramedics attended multiple heat-related incidents across the city yesterday.", "boundary_negative_news_emergency"),
    (0, "Children in outdoor sports were pulled from activities during the heat warning.", "boundary_negative_institutional_response"),
    (0, "Social workers checked on elderly and isolated residents during the heatwave.", "boundary_negative_community_third_party"),
    (0, "Heat-related hospital admissions rose sharply over the bank holiday weekend.", "boundary_negative_news_health_report"),
    (0, "Local charities are distributing water and fans to low-income households.", "boundary_negative_news_service"),
    (0, "The heat advisory prompted widespread closures of outdoor sporting facilities.", "boundary_negative_institutional_response"),
    (0, "Rough sleepers were offered emergency heat refuge beds during the hot spell.", "boundary_negative_news_service"),
    (0, "Animal welfare groups urged owners to keep pets inside during the heatwave.", "boundary_negative_news_advisory"),
    (0, "City swimming pools extended their hours during the extreme heat period.", "boundary_negative_institutional_response"),
    (0, "Public libraries became informal cooling refuges during the heat emergency.", "boundary_negative_news_service"),
    (0, "Heat-stroke deaths among migrant farmworkers sparked an official investigation.", "boundary_negative_news_mortality"),
    (0, "Firefighters battled multiple blazes sparked by the extreme heat conditions.", "boundary_negative_news_emergency"),
    (0, "Community groups organised welfare checks for isolated elderly neighbours.", "boundary_negative_community_third_party"),
    (0, "Public health officials recorded a spike in heat-related emergency room visits.", "boundary_negative_news_health_report"),
    (0, "Outdoor workers were provided sunscreen and shade tents under new regulations.", "boundary_negative_news_welfare_policy"),
    (0, "Heat-related school absences reached their highest level in a decade.", "boundary_negative_institutional_response"),
    (0, "The city opened 50 cooling stations after receiving 200 heat emergency calls.", "boundary_negative_news_emergency"),
    (0, "Mortality among the elderly was higher than seasonal norms during the heatwave.", "boundary_negative_news_mortality"),
    (0, "Social housing residents received portable fans from the council during the heat.", "boundary_negative_news_welfare_policy"),
    (0, "Local government issued food safety warnings as heat spoiled outdoor event produce.", "boundary_negative_news_advisory"),
    (0, "Animal rescue centres reported a surge in heat-stressed pets during the heatwave.", "boundary_negative_news_third_party"),
    (0, "Residents in the affected area were advised to avoid unnecessary outdoor activity.", "boundary_negative_news_advisory"),
    (0, "Third party contractors halted outdoor construction work due to the heat warning.", "boundary_negative_institutional_response"),
    (0, "Outreach workers distributed bottled water to rough sleepers during the heat spell.", "boundary_negative_community_third_party"),
    (0, "Transport workers were rotated off outdoor duties more frequently during peak heat.", "boundary_negative_news_welfare_policy"),
    (0, "Medical examiner confirmed four heat-related deaths over the long weekend.", "boundary_negative_news_mortality"),
    (0, "Neighbourhood associations arranged shaded rest areas for residents during heat.", "boundary_negative_community_third_party"),
    (0, "Heat emergency helpline received a record number of calls in a single day.", "boundary_negative_news_emergency"),
    (0, "Sports governing body suspended all outdoor competitions for three days of heat.", "boundary_negative_institutional_response"),
    (0, "Welfare checks were conducted on over 2,000 vulnerable residents during the heat.", "boundary_negative_news_welfare_policy"),
]

# All sentences combined — 620 unique entries, no sentence repeated
ALL_EXAMPLES = (
    POSITIVE_PHYSIOLOGICAL        # 60
    + POSITIVE_PSYCHOLOGICAL      # 50
    + POSITIVE_COPING             # 60
    + POSITIVE_AMBIENT_PERSONAL   # 60
    + NEGATIVE_POLICY             # 60
    + NEGATIVE_METAPHORICAL       # 60
    + NEGATIVE_PRODUCT_MEDIA      # 60
    + NEGATIVE_INDOOR_FOOD        # 50
    + NEGATIVE_PROPER_NOUN        # 40
    + NEGATIVE_SPAM               # 40
    + BOUNDARY_POSITIVE           # 40
    + BOUNDARY_NEGATIVE           # 40
)
# Total: 620 unique sentences

# ---------------------------------------------------------------------------
# Seasonal weighting
# The real corpus shows northern-hemisphere summer (Jun–Aug) has the highest
# positive rate.  We reproduce this signal in the synthetic demo by assigning
# months to examples with a label-aware, hemisphere-aware probability.
# ---------------------------------------------------------------------------

# Four seasons defined for both hemispheres.
# Northern hemisphere (study period Mar 2022 – Feb 2023):
#   Spring: Mar–May 2022   Summer: Jun–Aug 2022
#   Fall:   Sep–Nov 2022   Winter: Dec 2022–Feb 2023
# Southern hemisphere seasons are inverted:
#   Spring: Sep–Nov 2022   Summer: Dec 2022–Feb 2023
#   Fall:   Mar–May 2022   Winter: Jun–Aug 2022
#
# Positive-label weights mirror the manuscript's reported seasonal ranking:
#   Summer > Spring > Fall > Winter
# (Supplementary Note S3: median HPII Summer >> Spring ≈ Fall > Winter)

SEASON_WEIGHTS = {
    "north": {
        # Spring (moderate-high)
        "2022-03": 1.8, "2022-04": 1.8, "2022-05": 1.8,
        # Summer (highest)
        "2022-06": 3.0, "2022-07": 3.0, "2022-08": 3.0,
        # Fall (moderate)
        "2022-09": 1.2, "2022-10": 1.2, "2022-11": 1.2,
        # Winter (lowest)
        "2022-12": 0.5, "2023-01": 0.5, "2023-02": 0.5,
    },
    "south": {
        # Fall for southern hemisphere (Mar–May = northern spring)
        "2022-03": 1.2, "2022-04": 1.2, "2022-05": 1.2,
        # Winter for southern hemisphere (Jun–Aug = northern summer)
        "2022-06": 0.5, "2022-07": 0.5, "2022-08": 0.5,
        # Spring for southern hemisphere (Sep–Nov = northern fall)
        "2022-09": 1.8, "2022-10": 1.8, "2022-11": 1.8,
        # Summer for southern hemisphere (Dec–Feb = northern winter)
        "2022-12": 3.0, "2023-01": 3.0, "2023-02": 3.0,
    },
}

ALL_MONTHS = [
    "2022-03", "2022-04", "2022-05", "2022-06",
    "2022-07", "2022-08", "2022-09", "2022-10",
    "2022-11", "2022-12", "2023-01", "2023-02",
]


def seasonal_month_weight(month: str, hemisphere: str, label: int) -> float:
    """Return a relative sampling weight for (month, hemisphere, label).

    Negative examples are distributed uniformly across months.
    Positive examples follow the four-season weighting so that the synthetic
    corpus reproduces the Summer > Spring > Fall > Winter HPII ranking
    reported in Supplementary Note S3.
    """
    if label == 0:
        return 1.0
    return SEASON_WEIGHTS.get(hemisphere, SEASON_WEIGHTS["north"]).get(month, 1.0)


def weighted_month_choice(rng: random.Random, hemisphere: str, label: int) -> str:
    weights = [seasonal_month_weight(m, hemisphere, label) for m in ALL_MONTHS]
    total = sum(weights)
    r = rng.random() * total
    cumulative = 0.0
    for month, w in zip(ALL_MONTHS, weights):
        cumulative += w
        if r <= cumulative:
            return month
    return ALL_MONTHS[-1]


def make_date(month_str: str, rng: random.Random) -> str:
    year, month = int(month_str[:4]), int(month_str[5:7])
    days_in_month = 28 if month == 2 else 30 if month in (4, 6, 9, 11) else 31
    return f"{year}-{month:02d}-{rng.randint(1, days_in_month):02d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="sample_data")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load city reference
    cities_csv = out_dir / "cities_reference.csv"
    if not cities_csv.exists():
        raise FileNotFoundError(
            f"City reference file not found: {cities_csv}\n"
            "Ensure cities_reference.csv is in the same output directory."
        )
    cities_df = pd.read_csv(cities_csv)
    city_records = cities_df.to_dict("records")

    # Use all unique examples without any repetition.
    # ALL_EXAMPLES contains exactly 620 unique sentences — no while-loop cycling needed.
    pool = list(ALL_EXAMPLES)
    rng.shuffle(pool)
    target_n = len(pool)  # 620

    rows = []
    city_cycle_idx = 0
    for i, (label, sentence, category) in enumerate(pool):
        # Assign city (round-robin, shuffled order)
        city = city_records[city_cycle_idx % len(city_records)]
        city_cycle_idx += 1

        month = weighted_month_choice(rng, city["hemisphere"], label)
        date = make_date(month, rng)

        rows.append({
            "tweet_id":         f"DEMO_{i+1:04d}",
            "sentence":         sentence,
            "label":            label,
            "city_id":          city["city_id"],
            "city_name":        city["city_name"],
            "continent":        city["continent"],
            "country":          city["country"],
            "country_code":     city["country_code"],
            "hemisphere":       city["hemisphere"],
            "lat":              city["lat"],
            "lon":              city["lon"],
            "month":            month,
            "date":             date,
            "example_category": category,
            "is_synthetic":     1,
            "original_split":   "train" if i < int(target_n * 0.7) else "test",
        })

    df = pd.DataFrame(rows)

    # Sort by date to produce a realistic temporal ordering
    df["_sort"] = pd.to_datetime(df["date"])
    df = df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)

    out_csv = out_dir / "demo_tweets.csv"
    df.to_csv(out_csv, index=False)

    label_counts = df["label"].value_counts().to_dict()
    cat_counts = df["example_category"].value_counts().to_dict()
    continent_counts = df["continent"].value_counts().to_dict()

    # Monthly positive rate (shows seasonal signal)
    monthly = df.groupby("month")["label"].mean().round(3).to_dict()

    meta = {
        "description": (
            "Synthetic demonstration dataset for the heat-perception pipeline. "
            "NOT derived from real tweets. NOT for scientific inference. "
            "Generated to satisfy Nature Communications code-availability requirements "
            "while complying with Twitter/X Developer Agreement restrictions."
        ),
        "seed": args.seed,
        "n_examples": len(df),
        "n_unique_sentences": len(set(s for _, s, _ in ALL_EXAMPLES)),
        "label_distribution": label_counts,
        "positive_rate_pct": round(label_counts.get(1, 0) / len(df) * 100, 1),
        "continent_distribution": continent_counts,
        "n_cities": int(df["city_id"].nunique()),
        "n_countries": int(df["country_code"].nunique()),
        "month_range": [ALL_MONTHS[0], ALL_MONTHS[-1]],
        "monthly_positive_rate": monthly,
        "category_counts": cat_counts,
        "note_on_performance": (
            "A BERT model trained on this 620-example corpus will NOT reproduce the "
            "performance metrics in the manuscript (which required 74,938 labelled tweets). "
            "See sample_outputs/monthly_test_metrics_reference.csv for the reference figures "
            "from the actual corpus."
        ),
    }

    with open(out_dir / "demo_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
