# SPDX-License-Identifier: GPL-3.0-or-later
"""Author and validate the 240-prompt H3-AnimeEngine v3.0 Golden Curriculum Matrix.

Generates `train/prompts_v3_veo_240.jsonl` strictly adhering 100% to the official
MiniMax-H3 Full-Reference (R2V) & T2VA specification:
  - 6-section exact order:
      1. subject_definitions
      2. summary
      3. retention_analysis
      4. detailed_description (350-500 English words; style opener BEFORE [Shot 1])
      5. overall_soundscape
      6. non_diegetic_music
  - Curriculum split across 240 unique prompts:
      * 168 (70.0%) Single-view Front Hero R2VA (`84` solo + `84` two-character duels)
      *  48 (20.0%) Front + Back `<Picture 1> + <Picture 2>` 360-deg anti-limb-dislocation R2VA
      *  24 (10.0%) Pure T2VA high-motion anime/2.5D action & kinematic anchors
  - Frame bucket split (matches V3_LONG_HORIZON_1000STEP_AND_LIMB_FIX_PLAN.md §4):
      *  80 x 22f (~0.92s)
      * 120 x 39f (~1.62s)
      *  40 x 73f (~3.04s)
  - Style anchor split:
      * 120 x [2d_cel_sakuga]
      * 120 x [25d_illust]
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


REQUIRED_HEADERS = (
    "subject_definitions:",
    "summary:",
    "retention_analysis:",
    "detailed_description:",
    "overall_soundscape:",
    "non_diegetic_music:",
)


@dataclass(frozen=True)
class CharProfile:
    slug: str
    short_noun: str
    native_style: str  # "[25d_illust]" or "[2d_cel_sakuga]"
    front_ref: str
    back_ref: str
    front_subject_clause: str
    dual_subject_clause: str
    inline_first_mention: str
    back_180_clause: str
    retention_front: str
    retention_dual: str
    anatomy_guard: str
    solo_moves: tuple[tuple[str, str, str, str], ...]  # (summary_act, action_body, foley, battle_line)
    duel_offense: tuple[tuple[str, str, str], ...]     # (summary_phrase, attack_clause, sfx)
    duel_defense: tuple[tuple[str, str, str], ...]     # (summary_phrase, parry_clause, sfx)
    spin_routines: tuple[tuple[str, str, str], ...]    # (summary_spin, spin_clause, landing_clause)


CHARACTERS: dict[str, CharProfile] = {
    "gpt_dragon": CharProfile(
        slug="gpt_dragon",
        short_noun="silver-white dragon girl",
        native_style="[25d_illust]",
        front_ref="veo_cast/gpt_dragon_front.png",
        back_ref="veo_cast/gpt_dragon_back.png",
        front_subject_clause=(
            "the silver-white dragon girl whose appearance comes from <Picture {pic}>: "
            "long flowing silver-white hair with a single curling ahoge, twin curved striated "
            "ivory-lavender dragon horns, luminous violet eyes, pointed elfin ears, a lavender "
            "rose-knot hair ornament with hanging tassels on the left side of her head, "
            "translucent lavender-white bat-like dragon wings sprouting from her mid-back, a "
            "thick pale-scaled dragon tail, a high-collared sleeveless white gown with detached "
            "ruffled sleeves, silver-violet rose-knot waist ornaments with diamond chains, a "
            "layered high-slit white chiffon skirt, and white ankle-strap high heels."
        ),
        dual_subject_clause=(
            "the silver-white dragon girl whose front appearance comes from <Picture 1> and "
            "whose back hair, dragon wings, scaled tail, and dress silhouette come from <Picture 2>: "
            "long flowing silver-white hair with a single curling ahoge, twin curved striated "
            "ivory-lavender dragon horns, luminous violet eyes, a left-side lavender rose-knot "
            "tassel ornament, symmetrical translucent lavender-white bat-like dragon wings, a "
            "thick pale-scaled dragon tail, a high-collared white gown with detached ruffled "
            "sleeves, diamond-chained rose-knot waist ornaments, a layered high-slit white "
            "chiffon skirt, and white ankle-strap high heels."
        ),
        inline_first_mention=(
            "the silver-white dragon girl with twin curved ivory-lavender horns, luminous violet "
            "eyes, translucent lavender-white dragon wings, thick pale-scaled dragon tail, and "
            "layered high-slit white gown"
        ),
        back_180_clause=(
            "revealing her full rear silhouette from <Picture 2>—waist-length cascading silver-white "
            "back curls, left-side lavender rose-knot tassel, symmetrical translucent lavender-white "
            "dragon wings spread taut from her mid-back, diamond-chained rear waist ribbons, and "
            "her single thick pale-scaled dragon tail whipping cleanly to counterbalance her torque"
        ),
        retention_front=(
            "her silver-white hair, curling ahoge, twin curved dragon horns, violet eyes, lavender "
            "rose-knot tassel ornament, translucent dragon wings, thick scaled dragon tail, layered "
            "high-slit white gown, and white ankle-strap heels remain identical to <Picture {pic}>."
        ),
        retention_dual=(
            "her front facial identity, violet eyes, twin curved horns, and high-slit white bodice "
            "from <Picture 1> together with her cascading silver-white back curls, symmetrical "
            "translucent dragon wings, rear diamond-chain ribbons, and thick pale-scaled tail from "
            "<Picture 2> stay unified on a single anatomically solid body across the full 360-degree turn."
        ),
        anatomy_guard="two slender arms, two long legs in white ankle-strap heels, two symmetrical mid-back dragon wings, and one scaled tail",
        solo_moves=(
            (
                "executes an explosive vertical dragon-wing ascent and cleaves a falling meteoric boulder with a crystalline violet claw arc",
                "flexes both knees cleanly and beats her translucent lavender-white dragon wings downward in a powerful synchronized stroke, launching vertically while her right arm slashes overhead to project a razor-sharp crescent of crystalline amethyst energy and her left hand stabilizes her poise beside her ribcage",
                "a forceful leathery double wingbeat, rushing updraft air, and a resonant amethyst energy slash",
                "Skyward fang, cleave the clouds!",
            ),
            (
                "skims low across shallow mirror water on outstretched dragon wings before snapping into a tail-whip shockwave",
                "glides horizontally inches above the ground on taut lavender-white dragon wings with both legs extended cleanly behind her in white ankle-strap heels, then plants her left foot firmly and whips her thick pale-scaled dragon tail forward in a crisp horizontal arc that unleashes a ring of violet draconic runes",
                "high-speed wing-wash skimming stone, a firm heel plant, and a heavy scaled tail shockwave",
                "Hold your ground before the dragon crest!",
            ),
            (
                "performs an airborne backward vault off a stone pillar while firing twin geometric dragon-breath lances",
                "pushes symmetrically off both white-heeled feet into an agile backward aerial vault, keeping both knees crisply aligned while extending both open palms forward to loose twin spiraling lances of translucent violet-white starfire as her layered chiffon skirt flares cleanly",
                "crisp heel push-off on carved stone, snapping white chiffon hems, and twin roaring starfire lances",
                "Twin astral breath, pierce straight through!",
            ),
            (
                "drops from high altitude into a three-point draconic shockwave landing and unfurls a dome of amethyst ward sigils",
                "folds her translucent lavender-white wings tightly for a steep controlled dive, then snaps both wings wide open to brake cleanly above the floor, touching both white heels and her right fingertips down in a poised crouch as twelve hexagonal violet ward sigils bloom around her",
                "steep whistling dive wind, sudden wide wing-brake snap, and twelve chiming hexagonal crystal wards",
                "Nothing crosses this sanctuary!",
            ),
        ),
        duel_offense=(
            (
                "rockets diagonally downward on beating dragon wings with a blazing violet claw-lance thrust",
                "beats both translucent lavender-white dragon wings in a sharp downward burst, tucking her left knee cleanly while driving her right hand forward sheathed in a rigid spear-like gauntlet of amethyst light as her thick pale-scaled tail streams straight behind for balance",
                "thundering dragon wingbeats and a piercing amethyst lance surge",
            ),
            (
                "unleashes a low sweeping dragon-tail crescent wave followed by a high-slit roundhouse kick",
                "pivots cleanly on her left white ankle-strap heel with strict two-leg balance, whipping her thick pale-scaled dragon tail low to launch a violet shockwave arc before snapping her right leg high in a razor-crisp roundhouse kick",
                "a heavy scaled tail sweep and a sharp sonic kick snap",
            ),
            (
                "ascends on spread wings and fires a fan of seven rigid crystalline dragon-scale blades",
                "hovers steadily on spread lavender-white wings with both feet poised cleanly beneath her layered white gown, sweeping her right arm across her chest to launch seven rigid diamond-faceted violet scale blades in a tight planar fan",
                "steady powerful wing-beats and crystalline scale-blades whistling through air",
            ),
        ),
        duel_defense=(
            (
                "interlocks a glowing hexagonal amethyst forearm ward and counters with a wing-gust blast",
                "plants both white-heeled feet shoulder-width apart without knee distortion, raising her right forearm wrapped in a crisp hexagonal amethyst barrier to deflect the strike squarely before snapping both dragon wings forward to drive back the shockwave",
                "a ringing crystalline barrier parry followed by an explosive twin-wing gust",
            ),
            (
                "vaults cleanly over the incoming strike on angled dragon wings and answers with a tail-crest slash",
                "angles her translucent dragon wings to vault cleanly upward with both legs tucked neatly above the attack vector, whipping her pale-scaled dragon tail downward in a crisp counter-arc of violet sparks",
                "swift upward wing-vault wind and a crackling violet tail-slash",
            ),
        ),
        spin_routines=(
            (
                "executes a full 360-degree airborne dragon-wing corkscrew kick before alighting gracefully",
                "leaps vertically on a powerful wingbeat and enters a continuous clockwise 360-degree airborne spin kick, rotating smoothly from her front face and bodice at zero degrees through a crisp ninety-degree side profile, then at one hundred eighty degrees",
                "completes the final rotation back to zero-degree front view, touching down softly on both white ankle-strap heels with knees gently flexed and wings poised wide",
            ),
            (
                "performs a high-speed 360-degree standing pivot tail-and-wing whirlwind ward",
                "anchors cleanly on the ball of her left white heel and whips into a rapid full 360-degree standing rotational sweep, turning continuously from her front view at zero degrees into profile and at one hundred eighty degrees",
                "brings her rotation cleanly back to face the lens at three hundred sixty degrees, planting her right heel into a balanced wide stance as violet runes orbit her waist",
            ),
        ),
    ),
    "kimi_star": CharProfile(
        slug="kimi_star",
        short_noun="celestial flute-playing star girl",
        native_style="[25d_illust]",
        front_ref="veo_cast/kimi_star_front.png",
        back_ref="veo_cast/kimi_star_back.png",
        front_subject_clause=(
            "the celestial musician girl whose appearance and silver transverse flute come from "
            "<Picture {pic}>: long cascading silver-white hair shading to pale ice-blue spiral curls, "
            "soft lavender-blue eyes, a white frilled headband adorned with a silver crescent moon "
            "and a left-side black K-monogram ribbon bow, a high-collared white ruffled blouse under "
            "a navy-blue buttoned corset bodice, a starry midnight-blue shoulder capelet printed with "
            "crescent moons, an asymmetrical high-low navy-and-white ruffled skirt printed with "
            "golden musical staves and piano-key trim, a black K-emblem book pouch at her hip, a "
            "rigid straight silver transverse flute, white stockings, and bowed navy high heels."
        ),
        dual_subject_clause=(
            "the celestial musician girl whose front appearance comes from <Picture 1> and whose "
            "back hair curls, starry capelet, and musical-staff dress silhouette come from <Picture 2>: "
            "cascading silver-white to pale ice-blue spiral curls, soft lavender-blue eyes, a white "
            "frilled headband with a silver crescent moon and left-side black K-monogram ribbon bow, "
            "a white ruffled blouse under a navy corset bodice, a starry midnight-blue capelet, an "
            "asymmetrical high-low navy-and-white ruffled skirt printed with golden musical staves "
            "and piano-key trim, a hip K-emblem pouch, a rigid straight silver transverse flute, "
            "white stockings, and bowed navy high heels."
        ),
        inline_first_mention=(
            "the silver-white-haired celestial musician girl wearing her crescent-moon frilled "
            "headband, starry midnight-blue capelet, musical-staff high-low navy skirt, and bowed "
            "navy high heels while holding her rigid straight silver transverse flute"
        ),
        back_180_clause=(
            "displaying her exact back view from <Picture 2>—the black K-ribbon bow with crystal "
            "drops beneath her white frilled headband, dense silver-to-ice-blue spiral curls, her "
            "starry midnight-blue shoulder capelet, and layered navy musical-staff back skirt ruffles "
            "flaring symmetrically above two straight white-stockinged legs"
        ),
        retention_front=(
            "her silver-to-ice-blue spiral curls, lavender-blue eyes, crescent-moon frilled headband, "
            "black K-ribbon bow, starry navy capelet, musical-staff and piano-key skirt, K-emblem "
            "pouch, rigid straight silver flute, and bowed navy heels remain identical to <Picture {pic}>."
        ),
        retention_dual=(
            "her front face, lavender-blue eyes, buttoned navy corset, and piano-key skirt trim from "
            "<Picture 1> together with her rear crystal-drop hair bow, voluminous silver-blue back "
            "curls, starry capelet, and layered musical-staff rear skirt from <Picture 2> remain "
            "seamlessly unified on one two-armed, two-legged body throughout the 360-degree rotation."
        ),
        anatomy_guard="two arms, two white-stockinged legs in bowed navy heels, and an unbending straight silver flute tube",
        solo_moves=(
            (
                "conducts a sweeping crescent sonic slash with her rigid silver flute across a starlit stage",
                "steps forward into a poised fencer lunge on her bowed navy heels, gripping her straight silver transverse flute firmly in her right hand like a conductor's rapier—its metallic tube staying perfectly straight and rigid—and sweeps a wide horizontal arc that projects a glowing golden five-line musical staff crowned with treble-clef shockwaves",
                "crisp heel clicks on polished stone, a pure harmonic flute resonance, and a slicing golden sonic wave",
                "Let the starlight nocturne ring clear!",
            ),
            (
                "leaps across floating piano-key platforms while spinning a protective ring of celestial notes",
                "springs lightly from her left foot onto her right across hovering luminous glass steps with crisp two-leg agility, raising her straight silver flute horizontally to her lips with both hands in authentic embouchure alignment to send shimmering sapphire eighth-note wards spiraling outward",
                "crystalline glass footfalls, clear silver flute notes, and chiming starlight glyphs",
                "Every measure strikes right on beat!",
            ),
            (
                "executes a brisk side-step evasion and thrusts a concentrated starlight harmonic beam from her flute tip",
                "glides laterally in a crisp chasse step that fans her asymmetrical piano-key and musical-staff skirt cleanly without leg tangling, then locks her stance and points her rigid straight silver flute directly forward to fire a focused beam of silver-blue astral light",
                "quick leather-heel glide, rustling layered petticoats, and a bright harmonic laser chime",
                "Crescendo, straight to the mark!",
            ),
            (
                "leaps backward onto a stone balustrade and calls down a shower of crystalline crescent-moon arpeggios",
                "kicks cleanly off both bowed navy heels into a light backward hop onto a carved balustrade, keeping knees and ankles anatomically crisp while twirling her straight silver flute once between nimble fingers before sweeping it downward to release discrete diamond-sharp crescent projectiles",
                "light heel touchdown on marble, deft metallic flute flourish, and cascading crystal chimes",
                "Moonlight cadence, fall!",
            ),
        ),
        duel_offense=(
            (
                "lunges forward to drive a golden musical-staff sonic ribbon from her rigid silver flute",
                "drives forward off her rear bowed navy heel in an athletic lunge, thrusting her unbending straight silver transverse flute cleanly with her right hand to launch a fast-coiling ribbon of golden five-line musical staves and sharp starlight notes",
                "swift heel lunge and a resonant harmonic sonic slash",
            ),
            (
                "vaults into the air and sweeps down twin crescent-moon soundblades with her silver flute",
                "leaps cleanly into mid-air with her starry midnight-blue capelet billowing symmetrically, slashing her rigid straight silver flute diagonally from shoulder to hip to project twin razor-thin silver crescent soundblades",
                "airy skirt snap and twin whistling harmonic soundblades",
            ),
            (
                "glides across the arena in a brisk waltz step while firing rapid staccato starlight darts",
                "glides sideways in crisp two-beat footwork on her white-stockinged legs, tapping her straight silver flute twice to fire a volley of discrete cobalt-and-gold staccato star darts",
                "nimble heeled waltz steps and crisp staccato chime darts",
            ),
        ),
        duel_defense=(
            (
                "raises her rigid silver flute two-handed to parry behind a glowing treble-clef harmonic shield",
                "plants both bowed navy heels squarely on the floor and holds her straight silver flute horizontally with both hands—zero shaft curvature—projecting a taut circular golden treble-clef barrier that shatters the incoming blow into musical sparks",
                "a pure bell-like harmonic parry and scattering golden sparks",
            ),
            (
                "sidesteps cleanly with a capelet flourish and counters with a crisp sonic ripple",
                "pivots ninety degrees on her front foot in a clean balletic evade as her piano-key skirt hem flares crisply, tapping the tip of her rigid silver flute against the air to send back a radial silver soundwave",
                "crisp shoe pivot and a ringing concentric acoustic pulse",
            ),
        ),
        spin_routines=(
            (
                "performs a graceful 360-degree celestial waltz pirouette while projecting a spiral of golden musical staves",
                "rises cleanly onto the ball of her left bowed navy heel and initiates a brisk clockwise 360-degree pirouette while holding her straight silver flute poised diagonally, turning smoothly from her front bodice at zero degrees through ninety-degree profile and at one hundred eighty degrees",
                "completes the full 360-degree turn back to her front stance facing camera, lowering her right heel smoothly as golden staff ribbons settle into glittering motes",
            ),
            (
                "executes an airborne 360-degree starlight leap-turn with her silver flute held rigid overhead",
                "springs upward from both feet into a crisp vertical 360-degree jump-turn with her rigid straight silver flute raised cleanly in her right hand, rotating continuously from front view into side view and at one hundred eighty degrees",
                "completes the final quarter-turn back to facing front at three hundred sixty degrees, landing symmetrically on both navy heels with zero limb drift",
            ),
        ),
    ),
    "claude_chan": CharProfile(
        slug="claude_chan",
        short_noun="copper-orange grimoire mage girl",
        native_style="[25d_illust]",
        front_ref="veo_cast/claude_chan_front.png",
        back_ref="veo_cast/claude_chan_back.png",
        front_subject_clause=(
            "the copper-orange-haired mage scholar whose appearance and dark hardcover grimoire come "
            "from <Picture {pic}>: very long wavy copper-orange hair falling past her hips, calm "
            "amber-orange eyes, a left-side orange-and-black sunflower hair ornament with black "
            "ribbons, a cream Victorian blouse with a high lace collar and sunflower throat brooch, "
            "a wide black corset waistband with gold chains over a tasseled burnt-orange front panel, "
            "layered cream, black, and orange Victorian skirt panels with gold floral embroidery and "
            "hip sunflower bows, cream lace-up heeled ankle boots, and a rigid gold-embossed "
            "dark-brown hardcover grimoire."
        ),
        dual_subject_clause=(
            "the copper-orange-haired mage scholar whose front appearance comes from <Picture 1> and "
            "whose back hair curtain, twin waist sunflower bows, and Victorian skirt silhouette come "
            "from <Picture 2>: hip-length wavy copper-orange hair, calm amber-orange eyes, a left-side "
            "sunflower hair ornament with black ribbons, a high-collared cream Victorian lace blouse, "
            "a black corset waistband over a tasseled burnt-orange panel, layered gold-embroidered "
            "cream, black, and orange skirt panels with rear sunflower tassel bows, cream lace-up "
            "heeled ankle boots, and a rigid dark-brown hardcover grimoire."
        ),
        inline_first_mention=(
            "the copper-orange-haired mage scholar wearing her sunflower hair ornament, high-collared "
            "cream-black-orange Victorian dress with gold floral embroidery, and cream lace-up ankle "
            "boots while cradling her rigid hardcover grimoire"
        ),
        back_180_clause=(
            "showing her complete rear design from <Picture 2>—her left sunflower hair ribbon, the "
            "dense hip-length curtain of individual wavy copper-orange hair strands, twin rear "
            "black-and-orange sunflower waist bows with swinging black tassels, and gold-embroidered "
            "layered skirt hems flaring cleanly above both cream lace-up boots"
        ),
        retention_front=(
            "her hip-length copper-orange hair, amber-orange eyes, sunflower hair ornament, cream "
            "lace collar, black corset waistband, tasseled burnt-orange panel, gold-embroidered "
            "cream-black-orange Victorian skirt, cream lace-up boots, and hardcover grimoire remain "
            "identical to <Picture {pic}>."
        ),
        retention_dual=(
            "her front facial features, amber-orange eyes, throat brooch, and tasseled front panel "
            "from <Picture 1> together with her hip-length copper-orange back hair curtain, rear "
            "sunflower tassel bows, and layered Victorian hem from <Picture 2> stay locked on one "
            "single anatomically accurate body across the 360-degree turn."
        ),
        anatomy_guard="two arms with crisp five-finger hand poses, two legs in cream lace-up heeled boots, and a rigid flat-planed hardcover grimoire",
        solo_moves=(
            (
                "snaps open her hardcover grimoire to launch a radial fan of blazing amber-gold parchment blades",
                "plants both cream lace-up heeled ankle boots in a balanced diagonal stance, holds her rigid dark-brown hardcover grimoire flat in her left palm as pages rifle open rapidly, and sweeps her right arm horizontally with five crisp extended fingers to fire nine rigid ember-gold rune blades",
                "rapid fluttering parchment pages, crisp boot heels bracing on stone, and nine whistling golden rune blades",
                "Chapter of solar iron, scatter and strike!",
            ),
            (
                "dashes laterally across the stone floor while erecting three interlocking octagonal solar rune shields",
                "sprints smoothly to the right with clean alternating knee-and-ankle stride mechanics beneath her gold-embroidered Victorian skirt, skids neatly to a halt on both cream boots, and thrusts her open hardcover grimoire forward to lock three rotating octagonal amber sigil barriers in mid-air",
                "brisk heeled boot steps, a clean stone skid, and resonant interlocking golden sigil chimes",
                "Aegis of the seventh folio, hold firm!",
            ),
            (
                "vaults backward onto an elevated dais and commands an ascending pillar of golden typographic runes",
                "pushes crisply off both feet in an athletic backward leap onto an elevated stone step while keeping her copper-orange hair streaming in distinct ribbons, cradling her grimoire against her left forearm and raising her right hand upward to ignite a vertical column of crisp amber-gold script",
                "crisp two-boot landing on marble, snapping Victorian skirt layers, and a roaring pillar of golden rune fire",
                "Rise in luminous script!",
            ),
            (
                "executes a low ducking pivot beneath a hazard and counters with twin orbiting bookmark lances",
                "drops smoothly into a poised low knee-bend without joint distortion, pivots cleanly on the balls of her cream lace-up boots while her hip tassels swing outward, and snaps her hardcover grimoire shut with one hand as twin rigid golden bookmark spears streak forward",
                "swift fabric rustle, a decisive hardcover book snap, and twin piercing golden lances",
                "Indexed and sealed!",
            ),
        ),
        duel_offense=(
            (
                "fans twelve rigid ember-orange rune pages from her open grimoire in a high-speed slicing barrage",
                "braces her rear cream lace-up boot firmly, holds her rigid hardcover grimoire open on her left palm, and slashes her right hand diagonally to launch twelve razor-edged amber-gold parchment blades in a fast staggered wave",
                "rapid snapping book pages and whistling golden blade slices",
            ),
            (
                "advances with a poised cross-step and projects a spinning solar astrolabe lance",
                "advances in a crisp cross-step on her heeled cream boots while her copper-orange hair flows cleanly behind her shoulders, extending her right index and middle fingers past her open grimoire to fire a rigid spinning spear of solar-gold geometry",
                "measured boot steps and a blazing solar spear hum",
            ),
            (
                "leaps upward to cast a descending cage of luminous copper-gold calligraphy chains",
                "springs cleanly into the air with both knees neatly flexed, holding her open grimoire chest-high as luminous copper-gold glyph ribbons lance downward in crisp straight vectors",
                "airy Victorian skirt flare and ringing metallic glyph chains",
            ),
        ),
        duel_defense=(
            (
                "deploys a crisp semi-circular fan of twelve glowing grimoire pages to split the incoming attack",
                "anchors both cream lace-up boots shoulder-width apart, cradling her open dark-brown grimoire in her left hand while fanning twelve rigid amber-gold rune pages into an unbroken semi-circular shield that parts the strike cleanly",
                "fluttering pages locking into a resonant crystalline shield chime",
            ),
            (
                "skids into an athletic lean-back dodge and counters with three sharp golden sigil cards",
                "locks both cream boot heels into a crisp lateral skid and arches her upper back athletically out of harm's way while flicking three rigid golden sigil cards from her right fingers",
                "sharp leather-sole skid on stone and three crisp card snaps",
            ),
        ),
        spin_routines=(
            (
                "executes a full 360-degree grimoire-casting spin that unfurls a circular ring of golden rune pages",
                "pivots briskly on the ball of her left cream lace-up boot into a continuous clockwise 360-degree turn while cradling her open hardcover grimoire at waist height, rotating smoothly from her front lace collar at zero degrees through profile and at one hundred eighty degrees",
                "completes the seamless 360-degree rotation back to facing the camera squarely, setting her right boot heel down firmly as twenty golden rune pages orbit cleanly around her",
            ),
            (
                "leaps into an airborne 360-degree Victorian skirt-flare vault before landing in a poised scholar stance",
                "launches straight upward off both cream boots into a clean airborne 360-degree vertical turn with her grimoire held securely against her chest, rotating continuously from zero-degree front view into ninety-degree profile and at one hundred eighty degrees",
                "finishes the full 360-degree turn facing front and touches down symmetrically on both cream lace-up boots with calm amber-orange eyes locked forward",
            ),
        ),
    ),
    "gemini_cat": CharProfile(
        slug="gemini_cat",
        short_noun="heterochromatic starlight cat-eared girl",
        native_style="[25d_illust]",
        front_ref="veo_cast/gemini_cat_front.png",
        back_ref="veo_cast/gemini_cat_back.png",
        front_subject_clause=(
            "the starlight cat-eared girl whose appearance comes from <Picture {pic}>: voluminous "
            "wavy periwinkle-violet hair shading to pastel rose-pink curls with a single curling "
            "ahoge, pointed cat ears with soft pink interiors, a prismatic four-pointed star hairpin "
            "on the left, heterochromatic eyes with a ruby-magenta right eye and a golden-amber left "
            "eye, a navy-and-white starlight idol dress with detached ruffled bell sleeves, trailing "
            "translucent starry sash ribbons tipped with four-pointed star pendants, a navy thigh "
            "strap on her right leg, a long fluffy violet-to-pink gradient cat tail, and white "
            "slouched-cuff star-emblem platform boots."
        ),
        dual_subject_clause=(
            "the starlight cat-eared girl whose front appearance comes from <Picture 1> and whose "
            "back hair curls, rear navy ribbon bow, cat tail, and dress silhouette come from <Picture 2>: "
            "voluminous wavy periwinkle-violet to pastel rose-pink hair with a curling ahoge, pointed "
            "cat ears, a left prismatic star hairpin and rear dark navy ribbon bow, heterochromatic "
            "ruby-magenta right eye and golden-amber left eye, a navy-and-white starlight dress with "
            "detached ruffled bell sleeves, star-pendant sash ribbons, a right-leg navy thigh strap, "
            "a fluffy violet-to-pink gradient cat tail, and white slouched-cuff platform boots."
        ),
        inline_first_mention=(
            "the starlight cat-eared girl with wavy periwinkle-to-pink hair, heterochromatic "
            "ruby-magenta and golden-amber eyes, right-leg thigh strap, fluffy gradient cat tail, "
            "and white cuffed star boots"
        ),
        back_180_clause=(
            "cleanly exposing her back view from <Picture 2>—the dark navy ribbon bow centered on her "
            "periwinkle-to-pastel-pink wavy back curls, her rear navy starlight skirt pleats, twin "
            "star-pendant sash ribbons, and her single fluffy violet-to-pink cat tail whipping tautly "
            "above two anatomically distinct legs"
        ),
        retention_front=(
            "her periwinkle-to-pink wavy hair, cat ears, prismatic star hairpin, heterochromatic "
            "ruby-magenta and golden-amber eyes, navy-and-white starlight dress, detached bell sleeves, "
            "star-pendant sashes, right-leg thigh strap, fluffy gradient cat tail, and white cuffed "
            "boots remain identical to <Picture {pic}>."
        ),
        retention_dual=(
            "her front facial identity, heterochromatic ruby-magenta and golden-amber eyes, and "
            "right-leg thigh strap from <Picture 1> together with her rear navy hair bow, cascading "
            "violet-to-pink back curls, star-pendant sashes, and fluffy gradient cat tail from "
            "<Picture 2> remain unified on one two-armed, two-legged body throughout the 360-degree spin."
        ),
        anatomy_guard="two arms in detached ruffled sleeves, two legs with the navy strap strictly on her right thigh, and one fluffy gradient cat tail",
        solo_moves=(
            (
                "sprints along a stone wall and launches an inverted starlight crescent flip-kick",
                "bounds three crisp steps along a vertical wall on her white slouched-cuff platform boots while her fluffy violet-to-pink cat tail streams behind as a dynamic rudder, then kicks off into an athletic mid-air arc and snaps her right leg—marked by its navy thigh strap—through a luminous cyan-and-magenta starlight crescent kick",
                "nimble boot steps on stone, a whip-fast aerial flip, and a chiming starlight crescent slash",
                "Twin-star kick, catch that!",
            ),
            (
                "performs a lightning-fast zig-zag cat-dash and fires twin heterochromatic astral beams from her palms",
                "darts left then right in razor-sharp zig-zag footwork with low athletic knee flexion on both white star boots, skids cleanly to a front-facing stance, and thrusts both hands forward side-by-side to fire intertwined ruby-magenta and golden-amber starbeams",
                "rapid rhythmic boot dashes, fluttering star-pendant ribbons, and twin shimmering astral beams",
                "Ruby and gold—double starlight burst!",
            ),
            (
                "vaults over a stone rail with a one-handed star-spring and lands in a poised feline crouch",
                "springs effortlessly upward, plants her left palm cleanly on a carved stone rail while tucking both knees symmetrically over the top without limb crossing, and drops into a balanced feline crouch on both white cuffed boots as diamond star motes scatter",
                "light palm vault on stone, jingling star pendants, and a cushioned two-boot feline landing",
                "Too slow to catch a starlight cat!",
            ),
            (
                "leaps into mid-air to spin twin four-pointed astral shuriken rings from both fingertips",
                "kicks straight upward with crisp two-leg symmetry, arching gracefully in mid-air while her fluffy violet-to-pink tail curls for poise and both hands flick twin rigid four-pointed prismatic star rings spinning cleanly forward",
                "crisp vertical leap, ringing crystal star blades, and airy ribbon snap",
                "Shining Gemini rings, fly!",
            ),
        ),
        duel_offense=(
            (
                "springs high for a diving diagonal starlight axe-kick trailed by prismatic sparks",
                "bounds high into the air with her pointed cat ears perked forward and her fluffy gradient cat tail arched cleanly, driving her right white-booted leg downward in a razor-sharp diagonal starlight kick",
                "swift feline leap and a blazing prismatic kick arc",
            ),
            (
                "zig-zags into close range to unleash twin ruby-and-gold astral claw slashes",
                "bursts forward in two crisp diagonal bounds on her white cuffed platform boots, slashing her right hand in a ruby-magenta arc and her left hand in a golden-amber arc with crisp five-finger claws of starlight",
                "rapid double-step dash and twin chiming astral claw slashes",
            ),
            (
                "launches twin spinning four-pointed constellation chakrams from an agile mid-air back-spring",
                "pushes off both feet into an athletic backward spring, extending both arms cleanly forward to hurl twin rigid four-pointed starlight chakrams along crisp converging trajectories",
                "clean boot push-off and whistling constellation chakrams",
            ),
        ),
        duel_defense=(
            (
                "executes an agile feline handspring evasion and snaps up a four-pointed starlight prism shield",
                "plants both palms cleanly for a crisp backward feline handspring onto both white platform boots, raising a glowing four-pointed prismatic star shield ahead of her chest",
                "nimble handspring landing and a bright prismatic shield chime",
            ),
            (
                "ducks smoothly beneath the blow with cat-tail counterbalance and counters with a low sweep kick",
                "drops cleanly into a low feline crouch with her fluffy violet-to-pink cat tail extended horizontally for balance, sweeping her right leg around in a crisp low starlight arc",
                "swift low crouch whoosh and a sparkling crescent sweep",
            ),
        ),
        spin_routines=(
            (
                "executes a full 360-degree airborne starlight spin kick before sticking a balanced two-boot landing",
                "bends both knees and launches straight upward into a continuous clockwise 360-degree airborne spin kick, rotating smoothly from her front torso and heterochromatic eyes at zero degrees into ninety-degree profile and at one hundred eighty degrees",
                "completes the final rotation back to zero-degree front view, landing symmetrically on both white slouched-cuff star boots with knees flexed and her cat tail curling neatly",
            ),
            (
                "performs a rapid 360-degree standing idol-fighter twirl that casts a circular constellation barrier",
                "pivots crisply on the ball of her left white star boot into a full 360-degree rotational twirl with both arms sweeping a luminous star ribbon, turning continuously from front view through profile and at one hundred eighty degrees",
                "arrives cleanly back at three hundred sixty degrees facing the lens, planting her right boot firmly with heterochromatic ruby-and-gold eyes sparkling",
            ),
        ),
    ),
    "grok_vampire": CharProfile(
        slug="grok_vampire",
        short_noun="blonde Gothic vampire axe girl",
        native_style="[2d_cel_sakuga]",
        front_ref="veo_cast/grok_vampire_front.png",
        back_ref="veo_cast/grok_vampire_back.png",
        front_subject_clause=(
            "the blonde Gothic vampire girl whose appearance and double-bladed Demonic Battle Axe "
            "come from <Picture {pic}>: golden-blonde hair bound into ultra-long twin-tails with "
            "black ribbon bows, small dark bat-wing hair clips, a black-and-gold crescent tiara, "
            "bright cerulean-blue eyes, a cute vampire fang, a high-collared black-and-crimson "
            "Gothic lolita dress with gold floral embroidery and waist chains, dark sheer thigh-high "
            "stockings with a gold garter ring, black-and-gold buckled Mary Jane heels, and a rigid "
            "long-shafted obsidian-and-bronze double-bladed Demonic Battle Axe with an X-shaped crossguard."
        ),
        dual_subject_clause=(
            "the blonde Gothic vampire girl whose front appearance and double-bladed Demonic Battle "
            "Axe come from <Picture 1> and whose back twin-tail hair, open-back strap bodice, and "
            "tiered Gothic dress silhouette come from <Picture 2>: ultra-long golden-blonde twin-tails "
            "anchored by black bows and bat-wing clips, a black-and-gold crescent tiara, bright "
            "cerulean-blue eyes, a small fang, a black-and-crimson gold-embroidered Gothic dress "
            "with a diagonal back strap and large rear waist bow, dark sheer stockings, buckled "
            "heels, and a rigid straight-shafted double-bladed Demonic Battle Axe."
        ),
        inline_first_mention=(
            "the golden-blonde twin-tailed Gothic vampire girl wearing her bat-wing clips, crescent "
            "tiara, and black-and-crimson gold-embroidered dress while gripping the rigid straight "
            "shaft of her double-bladed Demonic Battle Axe"
        ),
        back_180_clause=(
            "revealing her exact rear silhouette from <Picture 2>—symmetrical bat-wing clips and "
            "black bows anchoring twin whipping golden-blonde twin-tails, her open upper back crossed "
            "by its diagonal black strap, and the large rear waist bow above multi-tiered black-and-crimson "
            "gold-embroidered ruffles and two straight dark-stockinged legs"
        ),
        retention_front=(
            "her ultra-long golden-blonde twin-tails, bat-wing clips, black-and-gold tiara, blue "
            "eyes, small fang, black-and-crimson gold-embroidered Gothic dress, dark sheer stockings, "
            "buckled heels, and rigid straight-shafted double-bladed Demonic Battle Axe remain "
            "identical to <Picture {pic}>."
        ),
        retention_dual=(
            "her front facial expression, blue eyes, fang, front bodice chains, and rigid double-bladed "
            "Demonic Battle Axe from <Picture 1> together with her rear twin-tails, diagonal open-back "
            "strap, large rear waist bow, and tiered Gothic skirt from <Picture 2> stay unified on a "
            "single two-armed, two-legged body with zero shaft bending across the 360-degree swing."
        ),
        anatomy_guard="two arms firmly gripping the straight axe handle, two dark-stockinged legs in buckled heels, and zero bending along the rigid metal axe shaft",
        solo_moves=(
            (
                "leaps high and slams her rigid double-bladed Demonic Battle Axe downward in a crimson shockwave cleave",
                "kicks powerfully off both buckled Mary Jane heels into a high vertical leap while hoisting her long-shafted obsidian-and-bronze double-bladed Demonic Battle Axe overhead with both hands—keeping the entire metal shaft perfectly straight and rigid—and drives the crescent blade downward to split the flagstones in a crisp fan of crimson-and-gold sparks",
                "explosive heel leap, heavy metallic axe whoosh, and a earth-shaking stone-splitting impact",
                "Kneel before my crimson cleave!",
            ),
            (
                "sprints forward dragging sparks from her rigid Demonic Axe before launching an upward crescent slash",
                "sprints aggressively forward with crisp alternating leg strides beneath her black-and-crimson ruffled skirt while trailing the lower blade tip of her rigid straight-shafted Demonic Battle Axe across the stone to throw discrete orange sparks, then plants her front heel and rips a clean diagonal upward slash",
                "rapid heeled sprint steps, metallic blade scraping bright sparks on stone, and a fierce upward cleave",
                "Carve right through the night!",
            ),
            (
                "pole-vaults athletically using the butt spike of her rigid Demonic Axe and delivers an airborne spin-slash",
                "plants the bronze butt-spike of her unbending straight Demonic Battle Axe into the stone, vaults cleanly upward with both dark-stockinged legs tucked neatly past the rigid vertical shaft, and re-grips with both hands in mid-air to unleash a crisp horizontal crimson energy wave",
                "sharp metal butt-spike plant on stone, whipping blonde twin-tails, and a roaring crimson slash wave",
                "Too slow—look above you!",
            ),
            (
                "executes a rapid cross-guard parry stance followed by a sweeping figure-eight twin-blade flourish",
                "widens her stance firmly on both buckled heels, snaps the X-shaped crossguard of her rigid straight Demonic Battle Axe across her chest to shatter an ember hazard, and weaves the heavy double-bladed polearm through a crisp two-handed figure-eight arc with locked wrist-and-elbow mechanics",
                "ringing bronze crossguard clang, twin slicing air whooshes, and crisp heel lock on stone",
                "Not a scratch on this blade!",
            ),
        ),
        duel_offense=(
            (
                "plunges from above swinging a two-handed vertical cleave with her rigid Demonic Battle Axe",
                "leaps high with her ultra-long golden-blonde twin-tails streaming cleanly behind her shoulders, gripping the unbending straight obsidian-and-bronze shaft of her double-bladed Demonic Battle Axe with both hands to drive a heavy vertical cleave",
                "whipping twin-tails and a heavy cleaving axe roar",
            ),
            (
                "charges in to rip a wide horizontal crimson crescent swing with her straight-shafted Demonic Axe",
                "drives off her rear buckled heel in a powerful low charge, torquing her waist cleanly while swinging her rigid straight-shafted double-bladed Demonic Battle Axe horizontally in a razor-sharp crimson arc",
                "forceful boot drive and a whistling crimson axe sweep",
            ),
            (
                "lunges forward with a two-handed spear-thrust of her Demonic Axe's top spike followed by a backhand slash",
                "lunges cleanly on her front dark-stockinged leg, thrusting the central top spike of her rigid straight Demonic Battle Axe along a dead-straight axis before snapping the twin crescent blades sideways",
                "sharp metallic thrust and a crisp backhand blade whistle",
            ),
        ),
        duel_defense=(
            (
                "braces her rigid straight Demonic Battle Axe diagonally in both hands for an ironclad crossguard parry",
                "plants both buckled Mary Jane heels firmly shoulder-width apart and braces the straight unbending shaft of her double-bladed Demonic Battle Axe diagonally across both hands, catching the strike squarely on the bronze X-crossguard in a shower of sparks",
                "a thunderous steel-on-bronze parry clang and scattering sparks",
            ),
            (
                "back-steps cleanly while batting the attack aside with the flat of her rigid Demonic Axe blade",
                "executes a crisp two-step backward retreat on her buckled heels without leg crossing, swinging the broad obsidian-and-bronze axe head cleanly across her front to bat the incoming force aside",
                "two sharp heel steps and a heavy metallic deflection clang",
            ),
        ),
        spin_routines=(
            (
                "executes a full 360-degree heavy whirlwind cleave with her rigid Demonic Battle Axe before locking stance",
                "plants her left buckled heel as a rock-solid pivot and unleashes a full clockwise 360-degree horizontal whirlwind swing with both hands locked onto the perfectly straight shaft of her double-bladed Demonic Battle Axe, rotating smoothly from her front bodice at zero degrees through profile and at one hundred eighty degrees",
                "completes the full 360-degree rotation back to facing camera squarely, planting her right buckled heel wide as the straight axe shaft locks diagonally at her side",
            ),
            (
                "leaps into a 360-degree airborne tornado axe-spin with streaming blonde twin-tails and lands cleanly",
                "launches upward off both feet into an athletic 360-degree vertical spin while holding her rigid straight Demonic Battle Axe extended cleanly across her torso, turning seamlessly from front view into side profile and at one hundred eighty degrees",
                "finishes the 360-degree turn facing front and slams both buckled heels symmetrically onto the stone floor with crisp knee absorption",
            ),
        ),
    ),
    "deepseek_maid": CharProfile(
        slug="deepseek_maid",
        short_noun="cobalt-blue ocean whale-maid girl",
        native_style="[2d_cel_sakuga]",
        front_ref="veo_cast/deepseek_maid_front.png",
        back_ref="veo_cast/deepseek_maid_back.png",
        front_subject_clause=(
            "the full-proportion whale-maid girl whose appearance comes from <Picture {pic}>: very "
            "long wavy cobalt-blue hair shading to pale sky-blue at the tips with a single curling "
            "ahoge, large bright sea-blue eyes, blue fin-shaped ears trimmed with white frills and "
            "blue bows, a white frilled maid headdress, a navy sailor-style maid dress with a white "
            "ruffled bodice, pearl-pinned navy collar bow, gold waist buttons, a white apron "
            "embroidered with a small blue whale, a gold-embroidered navy skirt over a white "
            "petticoat, white thigh-high stockings with pale blue whale-tail tops, navy Mary Jane "
            "shoes with blue bows, and a large blue whale tail ending in a two-lobed fluke."
        ),
        dual_subject_clause=(
            "the full-proportion whale-maid girl whose front appearance comes from <Picture 1> and "
            "whose back hair gradient, rear apron bow, whale tail, and maid skirt silhouette come "
            "from <Picture 2>: long wavy cobalt-to-sky-blue hair with a curling ahoge, bright "
            "sea-blue eyes, white-frilled blue fin ears, a white frilled maid headdress, a navy "
            "sailor-style maid dress with pearl-pinned collar bow, whale-embroidered white apron "
            "tied in a crisp white rear bow, gold-embroidered navy skirt, white whale-motif "
            "thigh-high stockings, navy Mary Jane shoes, and a large two-lobed blue whale tail."
        ),
        inline_first_mention=(
            "the cobalt-blue whale-maid girl with white-frilled fin ears, navy sailor maid dress, "
            "whale-embroidered white apron, white thigh-high stockings, and large two-lobed blue "
            "whale tail"
        ),
        back_180_clause=(
            "displaying her clean back view from <Picture 2>—her white frilled headdress, symmetrical "
            "blue fin ears, cascading cobalt-to-sky-blue wavy hair tips, crisp white rear apron bow "
            "over her gold-embroidered navy skirt, and her single thick two-lobed blue whale tail "
            "sweeping cleanly above both white-stockinged legs"
        ),
        retention_front=(
            "her cobalt-to-sky-blue wavy hair, curling ahoge, frilled fin ears, maid headdress, "
            "pearl-pinned collar bow, navy sailor maid dress, whale-embroidered white apron, "
            "whale-motif white thigh-high stockings, navy Mary Jane shoes, and two-lobed blue whale "
            "tail remain identical to <Picture {pic}>."
        ),
        retention_dual=(
            "her front face, sea-blue eyes, ruffled white bodice, and whale-emblem apron from "
            "<Picture 1> together with her cobalt-to-sky-blue back hair waves, white rear apron bow, "
            "and thick two-lobed blue whale tail from <Picture 2> remain unified on a single "
            "two-armed, two-legged body throughout the 360-degree turn."
        ),
        anatomy_guard="two arms with crisp five-finger open palms, two full-proportion legs in white thigh-high stockings, and one two-lobed blue whale tail",
        solo_moves=(
            (
                "hydro-skates at high speed across mirror water and carves a pressurized cyan crescent wave",
                "glides forward like a figure skater on shimmering cyan hydro-rings beneath her navy Mary Jane shoes with crisp two-leg stride mechanics, then snaps into a sharp sideways hockey-stop skid while sweeping her large two-lobed blue whale tail forward to launch a razor-sharp crescent wave of sparkling turquoise water",
                "high-speed hydro-skating hiss, sharp water-carving skid, and a surging turquoise wave slash",
                "Tidal service, coming right through!",
            ),
            (
                "springs off a floating water ring into a forward somersault and drives a spiraling hydro-lance",
                "kicks cleanly off a hovering azure water ring with both navy Mary Jane shoes, tucks neatly through a crisp forward aerial flip without limb tangling, and thrusts her right open palm—all five fingers sharply resolved—to fire a tight spiraling lance of sapphire water",
                "crisp splash push-off, rushing aerial flip wind, and a pressurized spiraling water lance",
                "Deep-sea spiral, strike true!",
            ),
            (
                "dances across wet flagstones with rapid footwork while weaving twin ribbons of crystalline sea-foam",
                "advances in nimble cross-stepping footwork on her bowed navy shoes while her gold-embroidered maid skirt and white petticoat flare crisply, sweeping both hands in opposing arcs to weave twin ribbons of suspended crystalline water beads",
                "rhythmic shoe steps on wet stone, swirling pressurized water ribbons, and chiming spray droplets",
                "Every droplet follows the tide!",
            ),
            (
                "vaults backward off a coral pillar and slams down a radial tidal dome shield",
                "pushes symmetrically off a stone column with both feet, arcs backward cleanly with her cobalt-to-sky-blue hair streaming in distinct cel-shaded locks, and touches down in a balanced low crouch while slapping her whale tail fluke once to raise a translucent azure water dome",
                "firm shoe push-off, resonant tail-fluke slap on water, and a humming azure hydro-dome",
                "Harbor shield, hold steady!",
            ),
        ),
        duel_offense=(
            (
                "skims low and whips her two-lobed blue whale tail to unleash a razor-thin cyan hydro-blade",
                "drops into an agile low lunge on her white-stockinged legs and whips her large two-lobed blue whale tail forward in a crisp horizontal cleave that projects a translucent cyan hydro-blade",
                "swift low lunge and a slicing pressurized water blade",
            ),
            (
                "dives diagonally from a hovering water ring with a spiraling sapphire palm-lance",
                "kicks explosively off an airborne azure water ring, extending her right five-fingered hand forward to propel a tight high-pressure sapphire water lance ahead of her dive",
                "splash launch and a roaring high-pressure hydro-lance",
            ),
            (
                "sprints laterally while flicking five crisp crystalline water daggers",
                "sprints briskly across the arena on her navy Mary Jane shoes with clean stride separation, sweeping her left hand wide to launch five diamond-sharp cyan water daggers",
                "rapid shoe steps and five whistling crystal water darts",
            ),
        ),
        duel_defense=(
            (
                "spins a taut vortex shield of turquoise water ahead of her open five-fingered palm",
                "plants both navy Mary Jane shoes firmly shoulder-width apart and extends her open left palm—every finger cleanly drawn—to spin a crisp circular turquoise water-vortex shield that splits the incoming strike into suspended spray beads",
                "whirling hydro-vortex hum and crisp scattering water droplets",
            ),
            (
                "hydro-slides smoothly sideways out of range and counters with a tail-fin splash arc",
                "glides sideways in an instant low hydro-slide on gleaming water rings without ankle distortion, snapping her blue whale tail fluke upward to send back a crisp arc of pressurized spray",
                "smooth hydro-slide swoosh and a sharp tail-fluke counter-splash",
            ),
        ),
        spin_routines=(
            (
                "executes an airborne 360-degree corkscrew hydro-spin with full maid-skirt and whale-tail flare",
                "launches straight upward off both navy Mary Jane shoes into a brisk clockwise 360-degree vertical spin, rotating smoothly from her front whale-embroidered apron at zero degrees into ninety-degree profile and at one hundred eighty degrees",
                "completes the full 360-degree turn back to facing the camera squarely, landing lightly on both navy shoes with knees flexed as her curling ahoge bounces once",
            ),
            (
                "performs a high-speed 360-degree figure-skater hydro-pivot across shallow mirror water",
                "glides into center frame and pivots cleanly on her left shoe into a fast 360-degree rotational hydro-sweep that carves a circular crown of water beads, turning continuously from front view into profile and at one hundred eighty degrees",
                "arrives cleanly back at zero-degree front orientation, setting her right shoe down into a poised maid curtsy-guard",
            ),
        ),
    ),
    "deepseek_chibi": CharProfile(
        slug="deepseek_chibi",
        short_noun="two-head-tall chibi whale-maid girl",
        native_style="[2d_cel_sakuga]",
        front_ref="veo_cast/deepseek_chibi_front.png",
        back_ref="veo_cast/deepseek_chibi_back.png",
        front_subject_clause=(
            "the cute two-head-tall chibi Q-version whale-maid girl whose compact chibi proportions "
            "and appearance come from <Picture {pic}>: a large expressive chibi head, short compact "
            "chibi arms with tiny clenched hands, short stubby chibi legs, voluminous cobalt-to-sky-blue "
            "hair with a big curling ahoge, white-frilled blue fin ears with a blue bow, a white "
            "frilled maid headband, cheerful smiling chibi eyes and open pink grin, a navy chibi "
            "maid dress with a white whale-emblem apron and tiny gold skirt bows, short white "
            "frilled ankle socks, tiny navy bowed shoes, and a plump two-lobed blue whale tail."
        ),
        dual_subject_clause=(
            "the cute two-head-tall chibi Q-version whale-maid girl whose front appearance comes "
            "from <Picture 1> and whose back hair curtain, plump whale tail, and chibi dress silhouette "
            "come from <Picture 2>: strict two-head-tall Q-version proportions with a "
            "large chibi head, short compact chibi arms, and short stubby chibi legs, cobalt-to-sky-blue "
            "hair with a big curling ahoge, white-frilled blue fin ears, a white frilled headband, "
            "a navy chibi maid dress with gold skirt bows and white whale-emblem apron, short white "
            "ankle socks, tiny navy shoes, and a plump two-lobed blue whale tail."
        ),
        inline_first_mention=(
            "the cute two-head-tall chibi whale-maid girl with her oversized curling ahoge, "
            "white-frilled fin ears, whale-emblem apron, compact stubby chibi limbs, and plump blue "
            "whale tail"
        ),
        back_180_clause=(
            "showing her exact Q-version rear silhouette from <Picture 2>—her oversized white frilled "
            "headband and fin ears, rounded cobalt-to-sky-blue back hair curtain, flared navy-and-white "
            "chibi skirt, plump blue whale tail curving right, and short stubby chibi legs in white "
            "ankle socks without adult leg elongation"
        ),
        retention_front=(
            "her strict two-head-tall Q-version body ratio, large chibi head, short stubby limbs, "
            "cobalt-to-sky-blue hair, big curling ahoge, frilled fin ears, whale-emblem apron, white "
            "ankle socks, tiny navy shoes, and plump blue whale tail remain identical to <Picture {pic}>."
        ),
        retention_dual=(
            "her strict two-head-tall Q-version proportions, cheerful chibi face, and whale apron "
            "from <Picture 1> together with her rounded cobalt-to-sky-blue back hair, chibi skirt "
            "hem, white ankle socks, and plump whale tail from <Picture 2> stay locked on one "
            "compact chibi body without adult limb stretching across the 360-degree hop-spin."
        ),
        anatomy_guard="strict two-head-tall chibi proportions with two short chibi arms, two stubby chibi legs, and zero adult leg stretching",
        solo_moves=(
            (
                "bounces joyfully across floating star-bubbles with lively chibi squash-and-stretch hops",
                "springs upward with lively anime chibi squash-and-stretch energy while strictly preserving her compact two-head-tall Q-version body ratio, pumping both tiny chibi fists cheerfully as her stubby legs hop cleanly from one shimmering cyan water bubble to the next and her plump blue whale tail wags briskly",
                "bouncy cartoon bubble pops, tiny pitter-patter shoe hops, and cheerful water splashes",
                "Boing, boing—full speed ahead!",
            ),
            (
                "dashes in a rapid pitter-patter chibi sprint and pops up a giant sparkling whale-bubble shield",
                "runs forward with ultra-fast comedic pitter-patter steps on her short stubby chibi legs in white frilled ankle socks, skids to a cute wide-footed stop without stretching her compact torso, and claps both tiny hands together to inflate a crisp iridescent cyan whale-shaped bubble ward",
                "rapid pitter-patter tiny footsteps, a cute shoe squeak, and a bright shimmering bubble pop",
                "Super whale bubble, protect!",
            ),
            (
                "performs an energetic chibi somersault hop and slaps down a ring of star-shaped splash droplets",
                "bends her short chibi knees and bounces high into a neat compact tuck-flip—her oversized curling ahoge and fin ears staying razor-sharp—before patting the puddle surface with her plump two-lobed blue whale tail to send discrete star-shaped cyan droplets flying",
                "springy chibi jump swoosh, playful tail-slap splash, and twinkling droplet chimes",
                "Splashy star ring, ta-da!",
            ),
            (
                "slides down a shimmering hydro-rainbow ribbon and sticks a triumphant double-fist chibi pose",
                "surfs exuberantly down a curving ribbon of translucent turquoise water on both tiny navy bowed shoes with her short chibi arms spread wide for balance, then hops cleanly onto the stone terrace and pumps both tiny fists skyward with a beaming grin",
                "sparkling water-slide whoosh, light two-shoe hop, and a cheerful chime flourish",
                "Perfect landing, ten out of ten!",
            ),
        ),
        duel_offense=(
            (
                "hops high and bats a volley of bouncy cyan star-bubbles with her plump chibi whale tail",
                "springs upward with compact two-head-tall chibi energy and wags her plump two-lobed blue whale tail briskly to bat three crisp glowing cyan star-bubbles forward",
                "bouncy chibi hop and playful popping star-bubbles",
            ),
            (
                "zips forward in a rapid stubby-foot dash to launch a sparkling mini whirlpool ring",
                "zips across the floor on rapid pitter-patter stubby legs while keeping her oversized chibi head and whale apron crisp, thrusting both tiny hands forward to send a neat spinning cyan whirlpool ring",
                "fast tiny footsteps and a fizzy spinning water ring",
            ),
        ),
        duel_defense=(
            (
                "pops inside a resilient iridescent bubble dome and bounces cleanly back onto both tiny shoes",
                "tucks her short chibi arms and stubby legs neatly as a round iridescent cyan bubble shield rebounds the incoming force harmlessly, bouncing her smoothly onto both tiny navy shoes",
                "a rubbery iridescent bubble boing and a neat two-shoe bounce",
            ),
            (
                "ducks with an adorable chibi hop-sidestep while her big ahoge springs cleanly back up",
                "hops nimbly sideways on both short stubby legs out of the strike path while her oversized curling ahoge ducks and springs comically upright with crisp linework",
                "nimble tiny hop and a cute springy ahoge bounce",
            ),
        ),
        spin_routines=(
            (
                "executes a joyful 360-degree airborne chibi star-hop spin while keeping strict Q-version proportions",
                "bounces straight upward off both tiny navy bowed shoes into a brisk clockwise 360-degree chibi jump-spin while pumping both short arms happily, rotating smoothly from her cheerful front chibi face at zero degrees into side profile and at one hundred eighty degrees",
                "completes the full 360-degree turn back to facing the camera with a beaming grin, landing neatly on both stubby chibi feet as her big ahoge wiggles",
            ),
            (
                "performs a rapid 360-degree chibi tail-whirl twirl on the spot surrounded by sparkling bubbles",
                "pivots briskly on one tiny navy shoe into a continuous 360-degree chibi twirl that sends discrete cyan bubbles spiraling outward, turning smoothly from front view through profile and at one hundred eighty degrees",
                "stops cleanly facing front at three hundred sixty degrees on both short white-socked legs in a proud double-fist pose",
            ),
        ),
    ),
}


ARENAS: tuple[tuple[str, str, str, str], ...] = (
    (
        "moonlit Gothic flooded chapel",
        "inside a vaulted Gothic stone chapel flooded ankle-deep with surging mirror-dark water where shockwave ripples, drifting silver mist, and fluttering altar banners react dynamically across foreground arches",
        "Sweeping moonlight beams shift past drifting storm clouds outside tall lancet windows, casting lively moving rim highlights across wet limestone ribs and spraying droplets.",
        "surging chapel water ripples, snapping altar banners, echoing stone acoustics, and brisk night wind sweeping high arches",
    ),
    (
        "floating celestial brass astrolabe terrace",
        "atop an open white-marble sky-observatory ringed by giant counter-rotating brass astrolabe hoops, tumbling twilight clouds, and streaking golden starlight motes sweeping in deep three-dimensional layers",
        "Luminous golden twilight and moving aurora ribbons glide across engraved zodiac marble while distant floating monoliths drift slowly through bright cloud canyons.",
        "high-altitude twilight wind, resonant turning brass astrolabe rings, and crisp open-sky marble acoustics",
    ),
    (
        "sunlit clockwork cathedral gallery",
        "along a riveted iron catwalk weaving between monumental turning brass clockwork gears, swinging giant pendulums, and drifting sunlit dust streams in layered foreground and background depth",
        "Warm late-afternoon sunbeams strobe rhythmically through spinning gear teeth onto iron trusses and polished bronze cogs with deep three-dimensional perspective.",
        "rhythmic grinding and ticking of massive turning brass gears, swinging iron pendulums, and warm draft air",
    ),
    (
        "crimson-maple shrine courtyard at dusk",
        "across a tiered mountain vermilion shrine courtyard swept by swirling gusts of crimson maple leaves, fluttering paper shide streamers, and flickering stone-lantern flames in deep multi-plane perspective",
        "Clear amber-violet dusk clouds race overhead while warm dancing lantern light plays dynamically across lacquered cedar torii columns and whirling red autumn foliage.",
        "gusting autumn mountain wind whirling crimson maple leaves, fluttering shrine streamers, and resonant flagstone acoustics",
    ),
    (
        "storm-lit coastal lighthouse parapet",
        "on the storm-lashed circular granite gallery of a cliffside lighthouse where a bright rotating lantern beam sweeps continuously through diagonal silver rain and towering ocean spray plumes",
        "Rushing slate storm clouds and the sweeping white searchlight carve constantly shifting specular highlights across drenched stone battlements and churning turquoise surf below.",
        "howling coastal gale wind, slanting rain pattering granite, booming surf plumes, and humming lighthouse optics",
    ),
    (
        "sunlit turquoise glacier ice floe",
        "across fractured drifting turquoise glacier ice floes surrounded by foaming sapphire polar waves, swirling diamond spindrift snow, and towering crystalline ice arches",
        "Brilliant polar sunlight dances through moving sea spray and faceted ice spires, sending crisp cyan caustics rippling across slick glacial planes.",
        "whistling polar wind driving ice crystals, cracking glacier ridges, and churning arctic surf",
    ),
    (
        "stained-glass grand royal archive",
        "within a soaring circular grand archive where floating glowing parchment pages, swaying brass chandeliers, and towering mahogany spiral staircases rise through multi-story stained-glass galleries",
        "Vivid ruby, sapphire, and emerald light shafts shift across drifting enchanted manuscripts and gleaming marble floors as pages whirl through the air currents.",
        "whirling parchment currents, gently creaking brass chandeliers, and soaring domed library resonance",
    ),
    (
        "ancient sky-island ruined colonnade",
        "amid soaring fluted white-marble columns on a wind-swept floating sky-island racing past billowing sunlit cumulus towers, streaming ivy vines, and cascading sky-waterfalls",
        "High-altitude azure sunlight cuts dramatically across rushing cloud banks and foreground broken arches, creating bold shifting light-and-shadow depth.",
        "rushing high-altitude sky wind, roaring distant sky-waterfall mist, and snapping banner cloth",
    ),
    (
        "bioluminescent crystal sea cavern",
        "across a sculpted limestone arch spanning a shimmering subterranean lagoon where glowing cyan-violet crystal clusters pulse gently and cascading cave waterfalls mist through foreground stalactites",
        "Dynamic bioluminescent turquoise water caustics ripple continuously up dark wet cavern vaults and across faceted quartz prisms.",
        "echoing cave waterfall cascades, swirling underground lagoon ripples, and humming crystalline acoustics",
    ),
    (
        "golden-hour Venetian stone canal bridge",
        "over a carved limestone canal bridge where gliding wooden gondolas, rippling amber water, fluttering silk festival pennants, and wheeling white seabirds animate the sunlit Renaissance skyline",
        "Warm golden-hour sunbeams glint continuously off moving canal waves and fluttered silk banners across layered foreground balustrades and distant bell towers.",
        "lapping canal wakes, fluttering silk festival pennants, distant seabirds, and crisp limestone acoustics",
    ),
    (
        "neon-trimmed cyber-shrine rooftop",
        "atop a rain-slicked slate skyscraper roof framed by a pulsing holographic cyan torii gate, streaking aerial sky-trams, drifting neon steam vents, and crisp rain ripples",
        "Vibrant shifting cyan, magenta, and amber reflections ripple dynamically across wet slate puddles while distant holographic billboards animate the skyline.",
        "high-rise night wind, distant passing sky-tram whoosh, pulsing electronic torii hum, and rain splashing slate",
    ),
    (
        "starlit obsidian mirror amphitheater",
        "inside a vast circular obsidian amphitheater ringed by floating ignited braziers, sweeping emerald-and-violet aurora curtains overhead, and swirling glowing dust ribbons across reflective black stone",
        "Undulating aurora ribbons and leaping brazier flames cast constantly moving multi-colored highlights across polished obsidian tiers and inlaid silver star maps.",
        "open-air night wind fanning roaring fire braziers, subtle celestial aurora resonance, and crisp stone echoes",
    ),
)


CAMERA_MOVES_1SHOT: tuple[str, ...] = (
    "The camera sweeps in a fast 150-degree orbital crane arc around the action, plunging from an elevated angle to a low worm's-eye view as foreground pillars and deep architecture whirl past in 3D parallax.",
    "An agile FPV chase camera dives diagonally beside the movement, weaving past foreground architectural edges and billowing particles before whipping around into a low heroic front-three-quarter reveal.",
    "The camera executes a low-angle crash-dolly push-in with a ten-degree Dutch-tilt roll on peak impact, then rockets upward in a rising spiral crane revealing the living 3D arena below.",
    "The camera swoops from a high oblique angle down to ankle height in a curved J-hook glide, swinging around the action axis so foreground debris and background landmarks shift continuously in depth.",
    "A kinetic gimbal camera surges forward on a curved arc through foreground sparks and drifting mist, banking around the clash before pulling into a soaring rising crane angle.",
    "The camera drops rapidly from a high crane vantage into a tight low counter-orbit sweeping 120 degrees across the motion arc, making towering arches and particles revolve behind sharp silhouettes.",
    "The camera accelerates on a curved backward-and-upward dolly spiral ahead of the charge, banking into the velocity vector as layered foreground columns and distant structures rush past.",
    "A fluid 3D arc-and-pedestal camera sweeps low beneath the leap apex, tilting sharply upward against the living sky before cresting alongside the descent into dynamic eye-level framing.",
)

CAMERA_MOVES_SHOT2: tuple[str, ...] = (
    "Continuing the rotational energy, the camera arcs 75 degrees around the landing while craning upward as foreground particles and moving light sweep across depth planes.",
    "Riding the recoil momentum, the camera executes a curved low-angle dolly push-in with a leveling roll against continuously shifting background clouds and architectural parallax.",
    "The camera sweeps from a low oblique angle into a poised medium-full orbital glide, keeping fluttering ribbons, crisp outlines, and living background motion alive through the final frame.",
    "Flowing out of the impact energy, the camera pulls backward along a curved rising crane path revealing deep 3D space as spinning sparks stream past the lens.",
    "A curved gimbal arc glides around the poised guard from three-quarter left toward front, exposing volumetric depth across foreground stone edges and moving highlights.",
    "The camera drops from mid-air tracking height into a low counter-arc as shockwave ripples and drifting embers radiate forward past the lens.",
)

MUSIC_SCORES: tuple[str, ...] = (
    "High-tempo symphonic anime action score featuring rapid staccato string ostinatos, tight taiko and snare hits, and a bright brass accent on peak impact.",
    "Driving neo-classical battle arrangement with virtuosic solo violin runs, harpsichord arpeggios, and punchy double-kick percussion building cleanly to a crisp cadence.",
    "Energetic hybrid orchestral-electronic track combining nimble pizzicato strings, sparkling celesta motifs, and crisp syncopated drums locked to the choreography.",
    "Brisk chamber-action score led by agile flute flourishes, rhythmic acoustic guitar strumming, and taut cello spiccato resolving on a clear harmonic chord.",
    "Up-tempo heroic fantasy score with soaring French horns, driving timpani rolls, and rapid piano octaves accentuating every parry and landing.",
    "Crisp modern sakuga percussion-and-strings cue pairing tight rimshots and bass synth pulses with soaring legato violin counter-melodies.",
)


T2VA_SCENES: tuple[dict, ...] = (
    {
        "style": "[2d_cel_sakuga]",
        "summary": "an original twin-blade samurai girl sprinting along a moonlit castle ridge and cleaving a falling iron lantern chain in mid-air",
        "char_desc": "an agile young female swordswoman with a high jet-black ponytail tied by a scarlet cord, sharp emerald eyes, a navy haori patterned with white crane feathers over pleated hakama trousers, and two straight-bladed steel katanas",
        "act1": "sprints swiftly up curved grey ceramic roof tiles with crisp alternating tabi-boot strides, leaps cleanly off the ridge gargoyle with both knees tucked symmetrically, and crosses both unbending straight steel katana blades in an X-slash that severs a swinging iron chain cleanly in a shower of orange sparks",
        "act2": "lands noiselessly on both feet along the opposite roof eave, flicking both straight katana blades outward to her sides without wrist or elbow distortion while her scarlet hair cord flutters",
        "line": "One clean cut—path cleared!",
        "foley": "rapid tabi footsteps on ceramic tiles, leaping wind, a sharp twin-steel ring, and crisp landing",
    },
    {
        "style": "[25d_illust]",
        "summary": "an astral spear-valkyrie hovering above sunlit sea cliffs and hurling a crystalline lance of golden light through storm clouds",
        "char_desc": "a poised valkyrie maiden with braided platinum-blonde hair, sapphire eyes, a silver-winged headband, white-and-cobalt armored battle dress, knee-high silver greaves, and a straight-shafted golden astral halberd",
        "act1": "rises into the sunlit sky on shimmering golden light feathers, bracing both armored legs cleanly in mid-air while drawing back her right arm to poise her perfectly straight golden astral halberd overhead before hurling a blazing beam-spear forward",
        "act2": "catches the returning straight halberd shaft cleanly in both gauntleted hands and settles onto a cliffside marble pedestal with balanced two-foot poise",
        "line": "Dawn spear, part the storm!",
        "foley": "ringing silver armor, soaring coastal wind, a roaring golden spear beam, and solid boot touchdown",
    },
    {
        "style": "[2d_cel_sakuga]",
        "summary": "two rival kendo club girls trading lightning-fast bamboo shinai parries and a decisive men strike across a sunlit wooden dojo",
        "char_desc": "two high-school kendo athletes in crisp indigo kendogi jackets, pleated hakama, and protective men Bogu masks—one wearing a red tasuki ribbon and the other a white tasuki ribbon—each gripping a straight bamboo shinai with both hands",
        "act1": "glide rapidly across polished cypress floorboards in authentic suriashi sliding footwork, clashing their straight rigid bamboo shinai blades three times at center court without shaft warping before the red-ribbon fencer explodes forward in a textbook overhead strike",
        "act2": "slide cleanly apart into poised chudan-no-kamae stances at opposite court lines, both shinai tips locked center-to-center as sunlit dust motes hang crisply in the air",
        "line": "Ippon—match point!",
        "foley": "squeaking bare feet sliding on cypress wood, sharp bamboo shinai cracks, and resonant dojo acoustics",
    },
    {
        "style": "[25d_illust]",
        "summary": "a crimson-robed shrine maiden performing an acrobatic naginata ribbon-dance amidst swirling sakura petals on a lantern-lit stage",
        "char_desc": "a graceful shrine maiden with long glossy raven hair tied in a white paper mizuhiki bow, warm ruby eyes, crisp white kosode sleeves, scarlet hakama, white tabi socks, straw sandals, and a long straight-shafted lacquered naginata polearm",
        "act1": "steps briskly onto a cedar kagura stage, spinning her long straight-shafted naginata cleanly around her shoulders with fluid two-handed grip transitions—zero pole bending—before leaping into a wide split-kick slash that parts drifting pink sakura petals along a luminous crimson arc",
        "act2": "alights symmetrically on both sandal-clad feet and locks the straight naginata vertically beside her right shoulder as warm paper lanterns illuminate every silk fold",
        "line": "May the spring blossoms dance in peace!",
        "foley": "crisp sandal steps on cedar planks, whistling lacquered polearm spin, and rustling sakura petals",
    },
    {
        "style": "[2d_cel_sakuga]",
        "summary": "a cyber-courier rollerblading down a neon overpass rail and vaulting a holographic barrier with a spinning kick",
        "char_desc": "an energetic teenage courier girl with short spiky teal hair, amber goggles pushed onto her forehead, a cropped yellow windbreaker, dark athletic shorts, knee pads, and glowing cyan inline skates",
        "act1": "grinds at high speed down a curved steel overpass handrail on both inline skates with knees bent athletically and arms poised wide for balance, then kicks explosively into a high aerial vault over a red holographic toll barrier while snapping a crisp mid-air scissor kick",
        "act2": "touches all eight glowing skate wheels cleanly back onto the wet asphalt lane, carving a sharp braking arc in a spray of discrete neon-lit water beads",
        "line": "Express delivery, zero delay!",
        "foley": "metallic skate rail grind, rushing highway air, electronic barrier hum, and crisp wheel carve on wet asphalt",
    },
    {
        "style": "[25d_illust]",
        "summary": "an ice-crystal archer girl skating backward across a frozen alpine lake and loosing three luminous frost arrows in mid-leap",
        "char_desc": "an elven frost-archer girl with silver-mint braided hair, crystal-blue eyes, a fur-trimmed cerulean tunic, white leggings, silver figure-skate blades, and a rigid geometric crystal recurve bow",
        "act1": "glides backward across mirror-clear alpine lake ice on crisp alternating skate edges, leaps high into a poised aerial arch with two straight legs clearly separated, and draws three glowing cyan frost arrows simultaneously against her rigid crystal bow before releasing them in a clean planar fan",
        "act2": "lands smoothly on a single outside skate edge before setting her second blade down into a balanced stop amid sparkling ice shavings",
        "line": "Three winter stars, fly true!",
        "foley": "crisp steel skate blades carving lake ice, taut bowstring twang, and three chiming frost arrows",
    },
    {
        "style": "[2d_cel_sakuga]",
        "summary": "a chibi clockwork-witch mascot hopping across giant floating pocket-watches and swinging a star-tipped wand",
        "char_desc": "a cute two-head-tall chibi witch girl with a giant pointed navy hat, oversized emerald eyes, fluffy peach curls, short stubby chibi arms and legs in striped stockings, and a straight golden star-wand",
        "act1": "hops energetically with crisp Q-version squash-and-stretch physics from one hovering brass pocket-watch face to the next—strictly preserving her compact two-head-tall chibi body—and swings her straight golden star-wand to ignite a ring of twinkling emerald gear glyphs",
        "act2": "bounces neatly onto a central golden watch hand on both tiny buckled shoes and raises her star-wand proudly with a bright chibi grin",
        "line": "Tick-tock magic, right on time!",
        "foley": "springy chibi shoe hops, ticking brass pocket-watches, and sparkling wand chimes",
    },
    {
        "style": "[25d_illust]",
        "summary": "two aerial mage knights clashing glowing runic longswords high above a sunlit cloud sea",
        "char_desc": "a silver-haired mage knight in white-and-azure coat-armor wielding a straight sapphire broadsword facing a crimson-haired rival knight in black-and-gold coat-armor wielding a straight ruby broadsword",
        "act1": "streak toward each other across brilliant sunlit cloud canyons on glowing boot thrusters, locking their two perfectly straight runic broadswords dead-center in a radial shockwave ring that parts nearby cloud wisps while all four arms and four legs remain anatomically crisp",
        "act2": "flip cleanly backward off the blade lock into hovering ready stances twenty paces apart, straight blades humming with azure and crimson runes",
        "line": "Your guard has gotten sharper!",
        "foley": "high-altitude jet-rune hum, thunderous twin-broadsword clash, and rushing sky wind",
    },
)


def style_opener(style_tag: str, extra_guard: str) -> str:
    if style_tag == "[2d_cel_sakuga]":
        return (
            "The target video is in [2d_cel_sakuga] crisp 2D Japanese keyframe sakuga animation style "
            "filmed at an ultra-high shutter speed with zero motion blur, razor-sharp cel-shaded ink "
            "outlines, clean flat-and-rim color separation, and clearly resolved eyes, fingers, "
            f"hair strands, and costume details at peak velocity. Strict anatomical solidity—{extra_guard}—"
            "and unyielding straight-line weapon rigidity are preserved across every frame."
        )
    return (
        "The target video is in [25d_illust] luminous 2.5D semi-impasto anime illustration animation "
        "style filmed at an ultra-high shutter speed with zero motion blur, crisp cel-and-painted "
        "linework, rich dimensional rim lighting, and crystal-clear eyes, fingers, individual hair "
        f"locks, and fabric embroidery at peak velocity. Strict anatomical solidity—{extra_guard}—"
        "and unyielding straight-line weapon rigidity are preserved across every frame."
    )


def count_words_ws(text: str) -> int:
    return len(text.split())


def count_words_regex(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def tune_word_budget(desc_body: str, style_tag: str) -> str:
    """Ensure detailed_description sits squarely inside [355, 495] on BOTH ws and regex counts."""
    pad_clauses = [
        "Every subtle fold of layered cloth, flying hair ribbon, footwear buckle, and specular eye catchlight remains crisply inked without temporal shimmer, double outlines, or frame drift.",
        "Foreground stone textures, airborne particle sparks, and distant architectural silhouettes maintain clean depth parallax and stable color contrast throughout the continuous action.",
        "Joint angles across shoulders, elbows, wrists, hips, knees, and ankles articulate along natural biomechanical arcs so every limb silhouette reads with unmistakable clarity at peak speed.",
        "Ambient rim light and cast shadows stay physically locked to the primary light direction across every gesture, foot plant, and cloth settle without sudden exposure jumps.",
        "Hand proportions preserve five distinct fingers per palm while footwear soles maintain firm grounded contact with the arena floor upon every takeoff and landing.",
    ]
    idx = 0
    while (count_words_ws(desc_body) < 360 or count_words_regex(desc_body) < 360) and idx < len(pad_clauses):
        cand = desc_body.rstrip() + " " + pad_clauses[idx]
        if count_words_regex(cand) > 492:
            break
        desc_body = cand
        idx += 1
    return desc_body


def article_for(noun_phrase: str) -> str:
    return "an" if noun_phrase[:1].lower() in "aeiou" else "a"


def cap_first(text: str) -> str:
    t = text.strip()
    return t[:1].upper() + t[1:] if t else t


def build_solo_prompt(
    char: CharProfile,
    style_tag: str,
    frames: int,
    char_idx: int,
    slot_idx: int,
) -> dict:
    move = char.solo_moves[slot_idx % len(char.solo_moves)]
    arena = ARENAS[(slot_idx + char_idx * 5) % len(ARENAS)]
    cam1 = CAMERA_MOVES_1SHOT[(slot_idx + char_idx * 2) % len(CAMERA_MOVES_1SHOT)]
    cam2 = CAMERA_MOVES_SHOT2[(slot_idx + char_idx) % len(CAMERA_MOVES_SHOT2)]
    music = MUSIC_SCORES[(slot_idx + char_idx * 3) % len(MUSIC_SCORES)]
    use_speech = ((slot_idx + char_idx) % 2 == 0)
    use_single_take_39 = (frames == 39 and (slot_idx % 2 == 0))

    dur_str = "0.9-second" if frames == 22 else ("1.6-second" if frames == 39 else "3.0-second")
    if frames == 22 or use_single_take_39:
        shots_listed = "[Shot 1]"
    elif frames == 39:
        shots_listed = "[Shot 1], [Shot 2]"
    else:
        shots_listed = "[Shot 1], [Shot 2], [Shot 3]"

    subj_def = f"<Subject 1> is {char.front_subject_clause.format(pic=1)}"
    summary = (
        f"[reference generation] The target video is a {dur_str} high-speed solo action showcase "
        f"set in {article_for(arena[0])} {arena[0]}, where <Subject 1> {move[0]}."
    )
    retention = (
        f"<Subject 1> (appears in {shots_listed}): fully_preserved - "
        f"{char.retention_front.format(pic=1)}"
    )

    opener = style_opener(style_tag, char.anatomy_guard)
    speech_clause = (
        f' Locking her gaze forward, <Subject 1> (S1) calls out in a clear, spirited youthful voice, <d>[English] {move[3]}</d>'
        if use_speech
        else " Her expression stays fiercely focused with crisp unblinking eye highlights and cleanly drawn lips as the physical resonance of her maneuver rings cleanly across the space."
    )

    if frames == 22 or use_single_take_39:
        desc = (
            f"{opener}\n"
            f"[Shot 1] A continuous unbroken low-angle medium-wide shot opens {arena[1]}. {arena[2]} "
            f"Positioned cleanly at center-left frame, <Subject 1>, {char.inline_first_mention}, "
            f"gathers clear preparatory weight on both feet and then {move[1]}. {cam1} "
            f"Sounds of {move[2]} synchronize directly with her peak extension as discrete energy sparks "
            f"and crisp airborne dust motes trace the exact trajectory of her movement.{speech_clause} "
            f"Without any cut, her momentum carries smoothly into a poised, balanced two-foot guard "
            f"while her hair strands, layered garments, and footwear contours settle crisply through the final frame."
        )
    elif frames == 39:
        desc = (
            f"{opener}\n"
            f"[Shot 1] A dynamic low-angle medium-wide shot opens {arena[1]}. {arena[2]} "
            f"Starting at center-left frame facing right, <Subject 1>, {char.inline_first_mention}, "
            f"pushes cleanly off her braced stance and {move[1]}. {cam1} Sounds of {move[2]} punctuate "
            f"the peak apex of her motion while individual hair locks and layered hems fan outward in direct physical reaction.\n"
            f"[Shot 2] At 00:00.920, matching her forward landing momentum from Shot 1 along the exact same camera axis without crossing the 180-degree line, "
            f"the shot cuts to a tighter eye-level medium-full angle as <Subject 1> absorbs her touchdown into a rock-solid two-foot stance "
            f"with the same rim lighting and settling sparks still drifting past her shoulders.{speech_clause} {cam2}"
        )
    else:
        desc = (
            f"{opener}\n"
            f"[Shot 1] A wide action shot opens {arena[1]}. {arena[2]} "
            f"At center-left facing right, <Subject 1>, {char.inline_first_mention}, plants her rear foot "
            f"and {move[1]}. {cam1}\n"
            f"[Shot 2] At 00:01.180, cutting on the physical release beat along the same camera axis, "
            f"a three-quarter medium shot follows <Subject 1> through her follow-through weight shift "
            f"as {move[2]} echoes across the arena and sparks stream outward.{speech_clause}\n"
            f"[Shot 3] At 00:02.180, continuing the deceleration under unchanged lighting, "
            f"<Subject 1> sticks a balanced two-foot landing as ribbons and hair settle cleanly. {cam2}"
        )

    desc = tune_word_budget(desc, style_tag)
    prompt_text = (
        f"subject_definitions:\n{subj_def}\n\n"
        f"summary:\n{summary}\n\n"
        f"retention_analysis:\n{retention}\n\n"
        f"detailed_description:\n{desc}\n\n"
        f"overall_soundscape:\n{cap_first(move[2])} layered over {arena[3]}.\n\n"
        f"non_diegetic_music:\n{music}\n"
    )
    return {
        "task": "ref2va",
        "prompt": prompt_text,
        "ref_image": char.front_ref,
        "ref_images": [char.front_ref],
        "frames": frames,
        "style_tag": style_tag,
        "mode": "solo_front",
        "characters": [char.slug],
    }


def build_duel_prompt(
    char_a: CharProfile,
    char_b: CharProfile,
    style_tag: str,
    frames: int,
    variant_idx: int,
) -> dict:
    off = char_a.duel_offense[variant_idx % len(char_a.duel_offense)]
    dfn = char_b.duel_defense[(variant_idx // 2) % len(char_b.duel_defense)]
    arena = ARENAS[(variant_idx * 7 + len(char_a.slug) + len(char_b.slug)) % len(ARENAS)]
    cam1 = CAMERA_MOVES_1SHOT[(variant_idx * 5 + 2) % len(CAMERA_MOVES_1SHOT)]
    cam2 = CAMERA_MOVES_SHOT2[(variant_idx * 3 + 1) % len(CAMERA_MOVES_SHOT2)]
    music = MUSIC_SCORES[(variant_idx * 2 + 1) % len(MUSIC_SCORES)]
    use_speech = (variant_idx % 2 == 1)
    use_single_take_39 = (frames == 39 and (variant_idx % 2 == 0))

    dur_str = "0.9-second" if frames == 22 else ("1.6-second" if frames == 39 else "3.0-second")
    if frames == 22 or use_single_take_39:
        shots_listed = "[Shot 1]"
    elif frames == 39:
        shots_listed = "[Shot 1], [Shot 2]"
    else:
        shots_listed = "[Shot 1], [Shot 2], [Shot 3]"

    subj_def = (
        f"<Subject 1> is {char_a.front_subject_clause.format(pic=1)}\n"
        f"<Subject 2> is {char_b.front_subject_clause.format(pic=2)}"
    )
    summary = (
        f"[reference generation] The target video is a {dur_str} two-character high-speed duel "
        f"in {article_for(arena[0])} {arena[0]}, where <Subject 1> {off[0]} while <Subject 2> {dfn[0]}."
    )
    retention = (
        f"<Subject 1> (appears in {shots_listed}): fully_preserved - {char_a.retention_front.format(pic=1)}\n"
        f"<Subject 2> (appears in {shots_listed}): fully_preserved - {char_b.retention_front.format(pic=2)}"
    )

    guard_pair = "exactly two arms and two legs on each of the two distinct fighters with zero inter-character limb merging"
    if "chibi" in (char_a.slug, char_b.slug):
        guard_pair += " and strict preservation of the two-head-tall chibi proportions alongside full-scale partner anatomy"
    opener = style_opener(style_tag, guard_pair)

    line_text = char_a.solo_moves[variant_idx % len(char_a.solo_moves)][3]
    speech_clause = (
        f' Across the clash line, <Subject 1> (S1) locks eyes with <Subject 2> and exclaims clearly, <d>[English] {line_text}</d>'
        if use_speech
        else " Both <Subject 1> and <Subject 2> hold sharp, focused expressions with crisp eye catchlights as the physical clash shockwave ripples cleanly outward."
    )

    if frames == 22 or use_single_take_39:
        desc = (
            f"{opener}\n"
            f"[Shot 1] A continuous unbroken low-angle medium-wide two-shot opens {arena[1]}. {arena[2]} "
            f"Advancing explosively from frame left toward frame right, <Subject 1>, {char_a.inline_first_mention}, "
            f"{off[1]}. Simultaneously anchored at frame right facing left with clear spatial separation, "
            f"<Subject 2>, {char_b.inline_first_mention}, reacts in real time and {dfn[1]}. "
            f"The sounds of {off[2]} collide squarely against {dfn[2]} at center frame, driving both fighters smoothly back one crisp stride "
            f"into poised stances without any cut. {cam1}{speech_clause}"
        )
    elif frames == 39:
        desc = (
            f"{opener}\n"
            f"[Shot 1] A medium-wide two-shot opens {arena[1]}. {arena[2]} "
            f"Charging from frame left toward right, <Subject 1>, {char_a.inline_first_mention}, {off[1]}. "
            f"Bracing at frame right facing left, <Subject 2>, {char_b.inline_first_mention}, {dfn[1]}. "
            f"{cam1} Sounds of {off[2]} collide squarely with {dfn[2]} at center in a radial spark burst.\n"
            f"[Shot 2] At 00:00.900, cutting on the exact recoil shockwave while keeping the 180-degree axis—"
            f"<Subject 1> on frame left and <Subject 2> on frame right—both fighters skid apart onto balanced "
            f"two-foot stances as sparks drift in the same light.{speech_clause} {cam2}"
        )
    else:
        desc = (
            f"{opener}\n"
            f"[Shot 1] A wide two-shot opens {arena[1]}, framing <Subject 1> at frame left and <Subject 2> at frame right. {arena[2]} "
            f"Driving from frame left, <Subject 1>, {char_a.inline_first_mention}, {off[1]}. {cam1}\n"
            f"[Shot 2] At 00:01.150, cutting on the incoming strike trajectory along the same camera axis, "
            f"a medium two-shot captures <Subject 2>, {char_b.inline_first_mention}, at frame right as she {dfn[1]}, "
            f"meeting the assault in crisp sparks with {off[2]} and {dfn[2]}.{speech_clause}\n"
            f"[Shot 3] At 00:02.180, matching the elastic rebound along the unchanged axis, "
            f"<Subject 1> lands cleanly at frame left and <Subject 2> settles at frame right in poised guards. {cam2}"
        )

    desc = tune_word_budget(desc, style_tag)
    prompt_text = (
        f"subject_definitions:\n{subj_def}\n\n"
        f"summary:\n{summary}\n\n"
        f"retention_analysis:\n{retention}\n\n"
        f"detailed_description:\n{desc}\n\n"
        f"overall_soundscape:\n{cap_first(off[2])} colliding with {dfn[2]}, echoing across {arena[3]}.\n\n"
        f"non_diegetic_music:\n{music}\n"
    )
    return {
        "task": "ref2va",
        "prompt": prompt_text,
        "ref_image": char_a.front_ref,
        "ref_images": [char_a.front_ref, char_b.front_ref],
        "frames": frames,
        "style_tag": style_tag,
        "mode": "duel_front",
        "characters": [char_a.slug, char_b.slug],
    }


def build_spin360_prompt(
    char: CharProfile,
    style_tag: str,
    frames: int,
    variant_idx: int,
) -> dict:
    spin = char.spin_routines[variant_idx % len(char.spin_routines)]
    arena = ARENAS[(variant_idx * 5 + 3 + len(char.slug)) % len(ARENAS)]
    cam1 = CAMERA_MOVES_1SHOT[(variant_idx * 4 + 3) % len(CAMERA_MOVES_1SHOT)]
    cam2 = CAMERA_MOVES_SHOT2[(variant_idx + 1) % len(CAMERA_MOVES_SHOT2)]
    music = MUSIC_SCORES[(variant_idx * 3) % len(MUSIC_SCORES)]
    use_speech = (variant_idx % 2 == 0)

    dur_str = "0.9-second" if frames == 22 else ("1.6-second" if frames == 39 else "3.0-second")
    # CRITICAL ANTI-JUMP-CUT RULE: 22f & 39f 360-degree spins stay 100% single-take [Shot 1] so the
    # 0->90->180->270->360 body rotation is learned continuously with zero mid-spin cut cheating!
    # 73f keeps the ENTIRE 360-deg spin unbroken in [Shot 1] and only cuts cleanly AFTER touchdown in [Shot 2].
    shots_listed = "[Shot 1]" if frames in (22, 39) else "[Shot 1], [Shot 2]"

    subj_def = f"<Subject 1> is {char.dual_subject_clause}"
    summary = (
        f"[reference generation] The target video is a {dur_str} anti-limb-dislocation 360-degree "
        f"rotational showcase in {article_for(arena[0])} {arena[0]}, where a single <Subject 1> {spin[0]}—smoothly "
        f"transitioning in one continuous turn from her <Picture 1> front appearance through her "
        f"<Picture 2> back view and back to front."
    )
    retention = f"<Subject 1> (appears in {shots_listed}): fully_preserved - {char.retention_dual}"

    opener = style_opener(
        style_tag,
        f"one unified body with {char.anatomy_guard} rotating smoothly through 360 degrees with zero extra limbs or torso-leg counter-twisting",
    )
    line_text = char.solo_moves[(variant_idx + 1) % len(char.solo_moves)][3]
    speech_clause = (
        f' Facing the lens squarely once her full turn settles, <Subject 1> (S1) smiles confidently and says, <d>[English] {line_text}</d>'
        if use_speech
        else " Facing the camera squarely after completing the full rotation, <Subject 1> holds a crisp poised expression with bright unblinking eye highlights."
    )

    if frames in (22, 39):
        desc = (
            f"{opener}\n"
            f"[Shot 1] A continuous unbroken centered medium-full shot opens {arena[1]}. {arena[2]} "
            f"Standing squarely in her zero-degree front orientation matching <Picture 1>, <Subject 1>, "
            f"{char.inline_first_mention}, bends both knees cleanly for launch preparation and {spin[1]}, "
            f"{char.back_180_clause}, then carries her unbroken angular momentum seamlessly past two hundred seventy degrees "
            f"until she {spin[2]} matching <Picture 1> with zero limb ghosting or mid-spin jump-cuts.{speech_clause} {cam1}"
        )
    else:
        desc = (
            f"{opener}\n"
            f"[Shot 1] A wide shot opens {arena[1]}, framing <Subject 1>, "
            f"{char.inline_first_mention}, poised in her zero-degree front view from <Picture 1>. {arena[2]} "
            f"Bending both knees cleanly, she performs the full unbroken 360-degree turn inside one continuous take: "
            f"she {spin[1]}, {char.back_180_clause}, and rotates past two hundred seventy degrees as she {spin[2]} "
            f"facing camera again with zero mid-spin cuts. {cam1}\n"
            f"[Shot 2] At 00:02.150, cutting cleanly after touchdown along the same frontal axis, "
            f"a medium-full hero angle holds on <Subject 1> matching <Picture 1> as hair and ribbons settle.{speech_clause} {cam2}"
        )

    desc = tune_word_budget(desc, style_tag)
    prompt_text = (
        f"subject_definitions:\n{subj_def}\n\n"
        f"summary:\n{summary}\n\n"
        f"retention_analysis:\n{retention}\n\n"
        f"detailed_description:\n{desc}\n\n"
        f"overall_soundscape:\nCrisp footwork push-off, airy 360-degree rotational fabric and hair whoosh, and balanced two-foot touchdown across {arena[3]}.\n\n"
        f"non_diegetic_music:\n{music}\n"
    )
    return {
        "task": "ref2va",
        "prompt": prompt_text,
        "ref_image": char.front_ref,
        "ref_images": [char.front_ref, char.back_ref],
        "frames": frames,
        "style_tag": style_tag,
        "mode": "spin360_front_back",
        "characters": [char.slug],
    }


def build_t2va_prompt(scene_idx: int, frames: int) -> dict:
    sc = T2VA_SCENES[scene_idx % len(T2VA_SCENES)]
    arena = ARENAS[(scene_idx * 5 + 4) % len(ARENAS)]
    style_tag = sc["style"]
    cam1 = CAMERA_MOVES_1SHOT[(scene_idx * 3) % len(CAMERA_MOVES_1SHOT)]
    cam2 = CAMERA_MOVES_SHOT2[(scene_idx * 2 + 1) % len(CAMERA_MOVES_SHOT2)]
    music = MUSIC_SCORES[(scene_idx + 3) % len(MUSIC_SCORES)]
    use_speech = (scene_idx % 2 == 0)
    use_single_take_39 = (frames == 39 and (scene_idx % 2 == 0))

    dur_str = "0.9-second" if frames == 22 else ("1.6-second" if frames == 39 else "3.0-second")
    summary = f"[text generation] A {dur_str} high-motion anime action clip set in {article_for(arena[0])} {arena[0]}, showing {sc['summary']}."
    opener = style_opener(
        style_tag,
        "strict two-arm and two-leg human biomechanics with zero extra limbs",
    )
    speech_clause = (
        f' Speaking in a clear, energetic voice, the lead fighter (S1) declares, <d>[English] {sc["line"]}</d>'
        if use_speech
        else " Facial features, unblinking eyes, and individual fingers stay tack-sharp as crisp physical sound effects punctuate the finish."
    )

    if frames == 22 or use_single_take_39:
        desc = (
            f"{opener}\n"
            f"[Shot 1] A continuous unbroken low-angle wide-medium action shot opens {arena[1]}, featuring {sc['char_desc']}. "
            f"{arena[2]} Launching smoothly out of a grounded preparatory crouch, the subject {sc['act1']}, "
            f"and carries that exact physical momentum cleanly forward without cuts until the subject {sc['act2']}. "
            f"Crisp audible {sc['foley']} synchronizes directly with every footstep and weapon sweep.{speech_clause} {cam1}"
        )
    elif frames == 39:
        desc = (
            f"{opener}\n"
            f"[Shot 1] A dynamic medium-wide shot opens {arena[1]}, introducing {sc['char_desc']}. "
            f"{arena[2]} Driving cleanly into motion from frame left toward frame right, the subject {sc['act1']} "
            f"accompanied by crisp {sc['foley']}. {cam1}\n"
            f"[Shot 2] At 00:00.900, matching the carry-through momentum from Shot 1 along the same side of the 180-degree camera axis, "
            f"the shot cuts to an eye-level medium-full angle as the subject {sc['act2']} under identical ambient lighting "
            f"with crisp anatomical poise.{speech_clause} {cam2}"
        )
    else:
        desc = (
            f"{opener}\n"
            f"[Shot 1] A wide architectural action shot establishes {arena[1]}, framing {sc['char_desc']}. "
            f"{arena[2]} Accelerating smoothly from a poised guard at center-left, the subject {sc['act1']}. {cam1}\n"
            f"[Shot 2] At 00:01.180, cutting directly on the peak kinetic beat of Shot 1 without crossing the camera axis, "
            f"a three-quarter medium tracking angle follows the immediate physical follow-through, flying sparks, and {sc['foley']} across the arena.\n"
            f"[Shot 3] At 00:02.180, carrying the deceleration smoothly forward under unchanged lighting, "
            f"the shot cuts to a balanced medium-full hero composition as the subject {sc['act2']}.{speech_clause} {cam2}"
        )

    desc = tune_word_budget(desc, style_tag)
    prompt_text = (
        f"subject_definitions:\nN/A (text-to-video)\n\n"
        f"summary:\n{summary}\n\n"
        f"retention_analysis:\nN/A\n\n"
        f"detailed_description:\n{desc}\n\n"
        f"overall_soundscape:\n{cap_first(sc['foley'])} layered over {arena[3]}.\n\n"
        f"non_diegetic_music:\n{music}\n"
    )
    return {
        "task": "t2va",
        "prompt": prompt_text,
        "ref_image": None,
        "ref_images": [],
        "frames": frames,
        "style_tag": style_tag,
        "mode": "t2va",
        "characters": [],
    }


def allocate_frames(n: int, n_22: int, n_39: int, n_73: int) -> list[int]:
    assert n == n_22 + n_39 + n_73
    raw = [39] * n_39 + [22] * n_22 + [73] * n_73
    # Interleave deterministically so every small slice has balanced frame buckets
    out = [0] * n
    stride = 7
    pos = 0
    for val in raw:
        while out[pos] != 0:
            pos = (pos + 1) % n
        out[pos] = val
        pos = (pos + stride) % n
    return out


def generate_240_curriculum() -> list[dict]:
    slugs = list(CHARACTERS.keys())  # 7 characters

    # 1. Solo Front Hero: 84 prompts (12 per character; 28x22f, 42x39f, 14x73f)
    # Style target across solo: 42 [2d_cel_sakuga] + 42 [25d_illust]
    solo_frames = allocate_frames(84, 28, 42, 14)
    solo_items: list[dict] = []
    for i in range(84):
        c_idx = i % 7
        m_idx = i // 7
        c = CHARACTERS[slugs[c_idx]]
        if c.slug in ("gpt_dragon", "kimi_star", "gemini_cat"):
            st = "[25d_illust]"
        elif c.slug in ("grok_vampire", "deepseek_maid", "deepseek_chibi"):
            st = "[2d_cel_sakuga]"
        else:
            st = "[25d_illust]" if m_idx < 6 else "[2d_cel_sakuga]"
        solo_items.append(build_solo_prompt(c, st, solo_frames[i], c_idx, m_idx))

    # 2. Duel Front Hero: 84 prompts (28x22f, 42x39f, 14x73f; exact 42 [2d_cel_sakuga] + 42 [25d_illust])
    full_slugs = ["gpt_dragon", "kimi_star", "claude_chan", "gemini_cat", "grok_vampire", "deepseek_maid"]
    pairs_full = [(full_slugs[a], full_slugs[b]) for a in range(len(full_slugs)) for b in range(a + 1, len(full_slugs))]
    assert len(pairs_full) == 15  # 15 * 5 = 75 full-proportion duels
    chibi_pairs = [(s, "deepseek_chibi") for s in full_slugs] + [("deepseek_chibi", s) for s in ("deepseek_maid", "gemini_cat", "kimi_star")]
    assert len(chibi_pairs) == 9  # 75 + 9 = 84 duels!

    duel_pair_schedule: list[tuple[str, str]] = []
    for rep in range(5):
        for p in pairs_full:
            duel_pair_schedule.append(p if rep % 2 == 0 else (p[1], p[0]))
    duel_pair_schedule.extend(chibi_pairs)
    assert len(duel_pair_schedule) == 84

    duel_frames = allocate_frames(84, 28, 42, 14)
    duel_items: list[dict] = []
    for i, (sa, sb) in enumerate(duel_pair_schedule):
        st = "[25d_illust]" if i % 2 == 0 else "[2d_cel_sakuga]"
        duel_items.append(build_duel_prompt(CHARACTERS[sa], CHARACTERS[sb], st, duel_frames[i], i))

    # 3. 360-deg Front+Back Spin: 48 prompts (16x22f, 24x39f, 8x73f; exact 24 [2d_cel_sakuga] + 24 [25d_illust])
    spin_frames = allocate_frames(48, 16, 24, 8)
    spin_items: list[dict] = []
    for i in range(48):
        c = CHARACTERS[slugs[i % 7]]
        st = "[25d_illust]" if i % 2 == 0 else "[2d_cel_sakuga]"
        spin_items.append(build_spin360_prompt(c, st, spin_frames[i], i))

    # 4. Pure T2VA: 24 prompts (8x22f, 12x39f, 4x73f; exact 12 [2d_cel_sakuga] + 12 [25d_illust])
    t2va_frames = allocate_frames(24, 8, 12, 4)
    t2va_items: list[dict] = []
    for i in range(24):
        t2va_items.append(build_t2va_prompt(i, t2va_frames[i]))

    # Interleave across 24 blocks of 10 (each block of 10 has: 3-4 solo, 3-4 duel, 2 spin360, 1 t2va)
    # so `--limit 20` or `--limit 40` immediately exercises all 7 characters, all modes, and all buckets!
    ordered: list[dict] = []
    s_idx = d_idx = r_idx = t_idx = 0
    for blk in range(24):
        n_solo = 4 if blk % 2 == 0 else 3
        n_duel = 3 if blk % 2 == 0 else 4
        for _ in range(n_solo):
            ordered.append(solo_items[s_idx])
            s_idx += 1
        for _ in range(n_duel):
            ordered.append(duel_items[d_idx])
            d_idx += 1
        for _ in range(2):
            ordered.append(spin_items[r_idx])
            r_idx += 1
        ordered.append(t2va_items[t_idx])
        t_idx += 1

    assert len(ordered) == 240
    for idx, rec in enumerate(ordered, 1):
        rec["id"] = f"v3veo_{idx:04d}"
        rec["seed"] = 20263000 + idx * 13
    return ordered


def extract_sections(prompt_text: str) -> dict[str, str]:
    positions: list[tuple[str, int]] = []
    for hdr in REQUIRED_HEADERS:
        m = re.search(rf"(?m)^{re.escape(hdr)}\s*$", prompt_text)
        if not m:
            raise ValueError(f"Missing required standalone section header: {hdr}")
        positions.append((hdr, m.start()))

    if [p[1] for p in positions] != sorted(p[1] for p in positions):
        raise ValueError(f"Section headers out of required order: {positions}")

    sections: dict[str, str] = {}
    for i, (hdr, start) in enumerate(positions):
        body_start = prompt_text.index("\n", start) + 1
        body_end = positions[i + 1][1] if i + 1 < len(positions) else len(prompt_text)
        sections[hdr[:-1]] = prompt_text[body_start:body_end].strip()
    return sections


def validate_all_entries(records: list[dict], input_root: Path) -> dict:
    if len(records) != 240:
        raise AssertionError(f"Expected 240 records, got {len(records)}")

    seen_ids: set[str] = set()
    seen_prompts: set[str] = set()
    mode_counts: dict[str, int] = {}
    frame_counts: dict[int, int] = {}
    style_counts: dict[str, int] = {}
    char_counts: dict[str, int] = {}
    word_counts_ws: list[int] = []
    word_counts_re: list[int] = []

    anti_picture_re = re.compile(r"(?m)^\s*<Picture\s+\d+>\s+is\b")
    speaker_in_ret_re = re.compile(r"\(\s*<?S\d+>?\s*\)")

    for i, rec in enumerate(records, 1):
        rid = rec["id"]
        if rid in seen_ids:
            raise AssertionError(f"Duplicate id: {rid}")
        seen_ids.add(rid)

        prompt = rec["prompt"]
        norm_p = " ".join(prompt.split())
        if norm_p in seen_prompts:
            raise AssertionError(f"Duplicate prompt text at {rid}")
        seen_prompts.add(norm_p)

        if anti_picture_re.search(prompt):
            raise AssertionError(f"{rid}: Forbidden `<Picture N> is ...` anti-pattern found!")
        if re.search(r"\(<S\d+>\)", prompt):
            raise AssertionError(f"{rid}: Non-standard `(<S1>)` speaker tag found; use `(S1)`!")
        if re.search(r"[\u4e00-\u9fff]", prompt):
            raise AssertionError(f"{rid}: Non-English CJK character found in prompt!")

        secs = extract_sections(prompt)
        task = rec["task"]
        refs = rec.get("ref_images") or []

        if task == "ref2va":
            if not secs["summary"].startswith("[reference generation]"):
                raise AssertionError(f"{rid}: R2VA summary must start with `[reference generation]`")
            if not refs or not rec.get("ref_image"):
                raise AssertionError(f"{rid}: R2VA entry missing ref_image / ref_images")
            if rec["ref_image"] != refs[0]:
                raise AssertionError(f"{rid}: ref_image must equal ref_images[0]")
            for rp in refs:
                if not rp.startswith("veo_cast/"):
                    raise AssertionError(f"{rid}: ref path `{rp}` must be inside `veo_cast/`")
                full_p = input_root / rp
                if not full_p.is_file():
                    raise AssertionError(f"{rid}: missing ref image on disk: {full_p}")
            for pic_idx in range(1, len(refs) + 1):
                if f"<Picture {pic_idx}>" not in secs["subject_definitions"]:
                    raise AssertionError(f"{rid}: <Picture {pic_idx}> missing in subject_definitions")
            if "fully_preserved" not in secs["retention_analysis"]:
                raise AssertionError(f"{rid}: retention_analysis must use `fully_preserved`")
            if speaker_in_ret_re.search(secs["retention_analysis"]):
                raise AssertionError(f"{rid}: forbidden `(S1)` inside retention_analysis")
            if rec["mode"] == "spin360_front_back":
                if (
                    "whose front appearance" not in secs["subject_definitions"]
                    or "from <Picture 1> and whose back" not in secs["subject_definitions"]
                    or "from <Picture 2>:" not in secs["subject_definitions"]
                ):
                    raise AssertionError(f"{rid}: spin360 must use official <Picture 1>+<Picture 2> single-Subject syntax")
                if "<Subject 2>" in prompt:
                    raise AssertionError(f"{rid}: spin360 must define ONLY a single <Subject 1>")
        elif task == "t2va":
            if not secs["summary"].startswith("[text generation]"):
                raise AssertionError(f"{rid}: T2VA summary must start with `[text generation]`")
            if rec.get("ref_image") is not None or refs:
                raise AssertionError(f"{rid}: T2VA must have ref_image=None and ref_images=[]")
        else:
            raise AssertionError(f"{rid}: unknown task {task}")

        desc = secs["detailed_description"]
        if "[Shot 1]" not in desc:
            raise AssertionError(f"{rid}: detailed_description missing `[Shot 1]`")
        pre_shot1 = desc.split("[Shot 1]", 1)[0].strip()
        if not pre_shot1 or (
            "[2d_cel_sakuga]" not in pre_shot1 and "[25d_illust]" not in pre_shot1
        ):
            raise AssertionError(f"{rid}: missing aesthetic style opener with style tag BEFORE `[Shot 1]`")
        if re.search(r"\[Shot 1\]\s+At\s+\d", desc):
            raise AssertionError(f"{rid}: `[Shot 1]` must NOT carry a timestamp")
        for shot_n in (2, 3):
            if f"[Shot {shot_n}]" in desc and not re.search(rf"\[Shot {shot_n}\]\s+At\s+00:0\d\.\d{{3}},", desc):
                raise AssertionError(f"{rid}: `[Shot {shot_n}]` must start with `At MM:SS.mmm,`")

        wc_ws = count_words_ws(desc)
        wc_re = count_words_regex(desc)
        if not (350 <= wc_ws <= 500 and 350 <= wc_re <= 500):
            raise AssertionError(
                f"{rid}: detailed_description word count out of [350, 500]: ws={wc_ws}, regex={wc_re}"
            )
        word_counts_ws.append(wc_ws)
        word_counts_re.append(wc_re)

        mode_counts[rec["mode"]] = mode_counts.get(rec["mode"], 0) + 1
        frame_counts[rec["frames"]] = frame_counts.get(rec["frames"], 0) + 1
        style_counts[rec["style_tag"]] = style_counts.get(rec["style_tag"], 0) + 1
        for ch in rec["characters"]:
            char_counts[ch] = char_counts.get(ch, 0) + 1

    return {
        "total": len(records),
        "modes": mode_counts,
        "frames": frame_counts,
        "styles": style_counts,
        "characters": char_counts,
        "words_ws_min_max": [min(word_counts_ws), max(word_counts_ws)],
        "words_re_min_max": [min(word_counts_re), max(word_counts_re)],
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="train/prompts_v3_veo_240.jsonl")
    ap.add_argument("--input-root", default="D:/comfyui/ComfyUI/input")
    args = ap.parse_args(argv)

    records = generate_240_curriculum()
    stats = validate_all_entries(records, Path(args.input_root))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[v3-240] Wrote & validated 240/240 prompts -> {out_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
