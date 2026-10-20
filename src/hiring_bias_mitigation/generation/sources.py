"""External CV and JD sources for the teacher's training pool. English only, and opt-in.

V1's pool has one source, Djinni, which is Ukrainian IT. This module adds English sources
beside it, so that the teacher also decides on non-IT occupations and on CVs written in other
styles. By default nothing changes. `PoolConfig.sources` defaults to `("djinni",)`, and
`pool.build_pool` calls into this module only for the other names.

Pool sources
------------
A generation config names pool sources. Each one maps onto the staged sources of
`data/external.py::SOURCES` that it reads, and those hold the licence and the publish flag:

``rjdb``
    RJDB *train* triples. Each gives the matched CV with its own JD and, with
    `rjdb_include_weakened`, also the weakened CV with the same JD. dev and test are the
    held-out qualification-sensitivity set, and this module never reads their rows.
``kaggle_linkedin``
    Kaggle resume-dataset CVs paired with LinkedIn postings by occupation. The rule is in
    `kaggle_occupation` and `CATEGORY_TITLE_KEYWORDS`.

What every external text goes through, in this order, and why
-------------------------------------------------------------
1. *Cleaning.* Cleaning runs first, so that every later filter judges the text the teacher
   will see. Only the edits each cleaner names are made. E-mails, phone numbers and URLs are
   removed from every external text, CV or JD. They are PII, and they usually spell a name.
2. *The Djinni length bounds* (`pool.MIN/MAX_*_CHARS`). They are the same bounds, so a CV
   from any source is an equally long read for the teacher.
3. *Protected markers* (CVs only): `pool._MARKERS["en"]` plus `EXTERNAL_EXTRA_MARKERS_EN`.
   The counterfactual construction assumes the base CV carries no attribute. A US resume
   that says "Honorable Discharge, US Navy" contradicts an injected "Civilian", and Djinni's
   list was written for Djinni and does not catch that.
4. *JD requirements* (JDs only). A JD is dropped when a protected attribute is a stated
   requirement or preference of the job (`JD_REQUIREMENT_PATTERNS`). Then the attribute is
   job-relevant, and an attribute-driven verdict is no longer bias. Ordinary EEO boilerplate
   ("without regard to religion ... protected veteran status") is kept. It is a statement
   about the employer and changes no requirement.
5. *The external holdout.* A row whose CV or JD copies a held-out text
   (`external.find_external_leaks`) is dropped and counted under the source it copies.
   `pool.build_pool` then asserts both holdouts on the frame it returns.

Ids
---
Every external id is `external.prefixed_id(slug, native_id)`, so none can collide with a Djinni
UUID. `candidate_id` is the train/validation split key (`dataset.train_val_split`). It must
therefore group CVs that are versions of one person. An RJDB matched CV and its weakened
rewrite are the same person, and both get `rjdb:<triple_id>`. Their `pair_id`s differ.
Kaggle near-duplicate clusters are reduced to one CV before pairing, so each Kaggle
`candidate_id` is one CV.

Experience matching
-------------------
Djinni pairs a CV with a JD only when the candidate's years meet the job's stated years.
None of these sources has a structured years field on both sides. RJDB states years only
inside its free text. The Kaggle CVs have no years field, and LinkedIn has a coarse
`formatted_experience_level` (76% filled) and no years. So external pairs are not
experience-matched. `candidate_years` and `job_required_years` are NaN, and that means
unknown, not zero. RJDB pairs are plausible by construction anyway: the JD and the matched
CV were written from the same seed. Kaggle-LinkedIn pairs are plausible by occupation only,
which is a weaker match than Djinni's.
"""

from __future__ import annotations

import itertools
import re
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import external as X
from ..utils.logging import get_logger
from . import pool as PL

log = get_logger(__name__)

#: The pool source a generation config names -> the staged sources (data/external.py) it reads.
#: "djinni" reads none: it is the V1 pool, loaded from the Hub by `pool.load_djinni`.
POOL_SOURCES: dict[str, tuple[str, ...]] = {
    "djinni": (),
    "rjdb": ("rjdb",),
    "kaggle_linkedin": ("kaggle_resume", "linkedin_postings"),
}
EXTERNAL_LANGS = ("en",)

#: How a missing or empty `source` value is reported. It is never publishable: a row that
#: lost its provenance cannot be shown to be Djinni.
MISSING_SOURCE = "<missing>"


# ---------------------------------------------------------------------------------------
# Protected markers the Djinni list misses
# ---------------------------------------------------------------------------------------

#: Cues that US-style resumes carry and `pool._MARKERS["en"]` does not catch. Each group has
#: a measured reason. The verifiers found about 15-18 Kaggle CVs with Navy, Air Force or
#: Marine service that pass `_MARKERS`. `_MARKERS` catches 9 of O*NET's 620 military job
#: titles. Kaggle and ResumeAtlas CVs carry date-of-birth, nationality and sex fields. RJDB
#: CVs say "She is proficient ...". Like `_MARKERS` the list is over-inclusive on purpose.
#: A false positive costs one CV, and a false negative puts a contradicted attribute into a
#: counterfactual set. Applied after cleaning, so a removed personal block cannot fire.
EXTERNAL_EXTRA_MARKERS_EN = [
    # Military service: branches, units, ranks, discharge papers, military-only programmes.
    # "Old Navy" is a clothing retailer, and Kaggle's APPAREL CVs work there.
    r"(?i)(?<!old )\b(?:navy|naval|air force|marine corps|marines|usmc|coast guard"
    r"|national guard)\b",
    r"(?i)\b(?:squadron|platoon|battalion|regiment|brigade|infantry|artillery|cavalry)\b",
    r"(?i)\b(?:special forces|special operations|green beret|paratrooper|rifleman|gunner)\b",
    r"(?i)\b(?:warfare|ordnance|munitions|tank commander|submarine|soldiers?)\b",
    r"(?i)\bcombat (?:medic|veteran|zone|arms|engineer|tours?|deployments?|operations|unit)\b",
    # Words from O*NET's military job titles (55-xxxx) that civilian CVs rarely use.
    # "Weapons", "mortar", "armored" and "countermeasures" are left out: in Kaggle CVs they
    # are martial arts, brickwork, armored-car cash pickups and spill prevention.
    r"(?i)\b(?:missiles?|reconnaissance|counter-?intelligence|aircrew\w*|air defen[cs]e"
    r"|torpedo\w*|howitzer|rocket systems?|gunnery|fire direction|armou?r (?:crewman|officer))\b",
    r"(?i)\b(?:enlisted|sergeant|corporal|lieutenant|petty officer|airman|seaman)\b",
    r"(?i)\b(?:warrant officer|(?:non-?)?commissioned officer|nco|ncoic)\b",
    r"(?i)\b(?:honou?rabl[ey] discharged?|dd[- ]?214|tour of duty|service ?member)\b",
    r"(?i)\b(?:west point|j?rotc|g\.?i\.? bill|tricare|department of defen[cs]e|dod)\b",
    r"(?i)\bfort (?:bragg|hood|benning|campbell|lewis|carson|drum|riley|bliss|gordon|stewart"
    r"|polk|knox|sill|jackson|leonard wood|meade|belvoir)\b",
    # Demographic fields and self-descriptions. "Good citizenship" and a CMMS named Passport
    # are not fields, so those words need a field shape.
    r"(?i)\b(?:date of birth|d\.o\.b|dob|birth ?date|place of birth|born (?:on|in))\b",
    r"(?i)\b(?:nationality|visa status|caste|dual citizenship|citizen of)\b",
    r"(?i)\bcitizenship\s*:|\bpassport\s*(?:no\b|number\b|#|:)",
    r"(?i)\b(?:gender|sex|age|religion|marital)\s*:",
    r"(?i)\b(?:father|mother|husband|wife|spouse)['’]?s name\b",
    r"(?i)\b\d{2}\s*(?:years? old|yrs? old|y/o)\b",
    # Gender cues. Pronouns count in lower and title case only ("HIM" is Health Information
    # Management), and gender-neutral forms such as "he/she" and "his or her" do not count.
    r"\b(?:Mr|Mrs|Ms|Miss)\.?\s+[A-Z]",
    r"(?<![/\w])(?<!\bor )(?:[Ss]he|[Hh]er|[Hh]ers|[Hh]erself|[Hh]e|[Hh]im|[Hh]is|[Hh]imself)\b"
    r"(?!\s*/\s*\w|\s+or\s+(?:[Ss]he|[Hh]er|[Hh]e|[Hh]is|[Hh]im)\b)",
    r"(?i)\b(?:sorority|fraternity|sororities|fraternities)\b",
    r"(?i)\b(?:maternity|paternity|pregnan\w*)\b",
    # Religious institutions, roles and practice. "Temple" is left out: in these CVs it is
    # Temple University, and "Baptist Temple" is caught by "baptist".
    r"(?i)\b(?:church(?:es)?|ministry|ministries|minister|pastor(?:al)?|chaplain\w*|clergy)\b",
    r"(?i)\b(?:priest|rabbi|imam|mosque|synagogue|parish|diocese|seminary)\b",
    r"(?i)\b(?:bible|biblical|gospel|worship|missionary|mission trips?|theolog\w*|divinity)\b",
    r"(?i)\b(?:catholic|protestant|baptist|methodist|lutheran|presbyterian|evangelical)\b",
    r"(?i)\b(?:pentecostal|mormon|lds|islam\w*|quran|koran|torah|kosher|halal|ramadan)\b",
]
_EXTRA_MARKERS = [re.compile(p) for p in EXTERNAL_EXTRA_MARKERS_EN]


def has_external_extra_marker(text: str) -> bool:
    text = str(text)
    return any(p.search(text) for p in _EXTRA_MARKERS)


def has_external_marker(text: str) -> bool:
    """The protected-marker test for an external CV: Djinni's list, then the extra cues."""
    return PL.has_protected_marker(text, "en") or has_external_extra_marker(text)


# ---------------------------------------------------------------------------------------
# JDs that make a protected attribute a requirement
# ---------------------------------------------------------------------------------------

#: A JD that matches any of these is dropped. Apostrophes may be typographic ("Veterans’
#: Preference"), as they are in the postings. The verifier counted these cues in LinkedIn
#: postings: veteran preference 1,015, security clearance 2,457, US citizenship 1,841,
#: minimum age 4,762 and religious employer 1,591. In such a job the attribute is
#: job-relevant, so the counterfactual premise breaks. A clearance posting may lawfully
#: prefer veterans, and a ministry may lawfully require a faith. EEO sentences are left
#: alone. None of these patterns matches "without regard to ... religion ... veteran
#: status" or "minorities, women, veterans ... are encouraged to apply".
JD_REQUIREMENT_PATTERNS: dict[str, str] = {
    "security_clearance": (
        r"(?i)\b(?:security|secret|top secret|dod|government|federal|active|interim) clearance\b"
        r"|\bclearance (?:is |will be )?required\b|\b(?:ts|top secret)\s*/\s*sci\b"
        r"|\btop secret\b|\bpublic trust\b|\bpolygraph\b"
        r"|\b(?:obtain|maintain|hold|possess)(?: and maintain)? an? (?:active )?"
        r"(?:security )?clearance\b"
    ),
    "veteran_preference": (
        r"(?i)\bveterans?['’]? preference\b|\bpreference (?:will be |is )?(?:given|extended) to "
        r"(?:qualified )?(?:veterans|military)\b|\bpreference for (?:qualified )?veterans\b"
        r"|\bveterans? (?:are |is )?preferred\b"
        r"|\bmilitary (?:experience|background|service) (?:is )?(?:preferred|required|a plus)\b"
        r"|\bprior military\b|\bformer military\b|\bmilitary spouses?\b|\bmilitary[- ]friendly\b"
        r"|\btransitioning (?:service ?members|military)\b|\bdd[- ]?214\b|\bskillbridge\b"
    ),
    "us_citizenship": (
        r"(?i)\b(?:must|required to|need to|needs to) (?:be|hold|have|possess) (?:a |an )?"
        r"(?:u\.?s\.?|united states|american) citizen(?:ship)?\b"
        r"|\b(?:u\.?s\.?|united states) citizen(?:ship)? (?:is |status is )?"
        r"(?:required|only|mandatory)\b|\bcitizenship (?:is )?required\b"
        r"|\brequires? (?:u\.?s\.?|united states) citizenship\b"
    ),
    "religious_employer": (
        r"(?i)\b(?:church(?:es)?|ministry|ministries|pastor(?:al)?|chaplain\w*|clergy|parish"
        r"|diocese|archdiocese|congregation|synagogue|mosque|seminary|faith[- ]based|biblical"
        r"|bible|gospel|worship|catholic|jesuit|lutheran|baptist|methodist|presbyterian"
        r"|evangelical|jesus|christ)\b|\bstatement of faith\b|\bchristian (?:school|college"
        r"|university|organi[sz]ation|ministry|values|faith|environment|camp|academy)\b"
    ),
    "age_limit": (
        r"(?i)\b(?:must|needs? to|has to|have to|required to) be (?:at least |a minimum of "
        r"|over |over the age of )?(?:1[6-9]|2[0-5])\b"
        r"|\b(?:1[6-9]|2[0-5])\s*(?:\+\s*)?(?:years?|yrs\.?)\s*(?:of age|old|or older)\b"
        r"|\b(?:minimum|min\.?|maximum) age\b|\bage (?:requirement|limit|restriction)s?\b"
        r"|\b(?:at least|over|under) the age of\b|\bof legal (?:drinking |working )?age\b"
    ),
    # "Except where sex is a bona fide occupational qualification" is EEO boilerplate, so a
    # BFOQ mention alone is not a requirement.
    "sex_requirement": (
        r"(?i)\b(?:female|male|women|men) (?:only|candidates only|applicants only)\b"
        r"|\bmust be (?:a )?(?:female|male|woman|man)\b"
    ),
}
_JD_REQUIREMENTS = {k: re.compile(v) for k, v in JD_REQUIREMENT_PATTERNS.items()}


def jd_requirements(text: str) -> list[str]:
    """Every protected-attribute requirement a JD states, by name; empty if none."""
    text = str(text)
    return [name for name, pattern in _JD_REQUIREMENTS.items() if pattern.search(text)]


# ---------------------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------------------

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_URL = re.compile(
    r"(?i)\b(?:https?://|www\.)\S+"
    r"|\b(?:linkedin|github|facebook|twitter|instagram|behance)\.com/\S*"
)
_PHONE = re.compile(r"(?<![\d-])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]\d{4}(?![\d-])")


def scrub_pii(text: str) -> str:
    """Removes e-mail addresses, URLs and phone numbers, and leaves the rest as it was.

    They are removed rather than replaced with a placeholder. A placeholder adds text that no
    real CV has, and the teacher would decide on that text. A label left without its value
    ("Email:") is harmless.
    """
    return _PHONE.sub("", _URL.sub("", _EMAIL.sub("", str(text))))


def collapse_whitespace(text: str) -> str:
    """One space for every run of horizontal whitespace, and at most one blank line in a row.

    The Kaggle CV text is padded with spaces: 18.7% of its characters are whitespace, and 47%
    of the CVs exceed the 6,000-character bound only because of that padding.
    """
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_kaggle_cv(text: str) -> str:
    return collapse_whitespace(scrub_pii(text))


def clean_jd(text: str) -> str:
    return scrub_pii(text).strip()


#: The first line of a CV section, i.e. the end of RJDB's personal block. "Education:" with up
#: to 40 more characters is a header. A longer line is a section written inline. The keyword
#: may be wrapped in "#", "**" or "__", so it ends at a non-letter, not at a word boundary.
_SECTION = re.compile(
    r"(?i)^[\W_]*(?:education|academic\w*|skills?|technical skills|key skills|core competenc\w*"
    r"|competenc\w*|(?:work |professional |relevant |employment )?experience|employment"
    r"|work history|career history|(?:professional |career |executive )?summary"
    r"|(?:career |professional )?objective|profile|about me|certific\w*|qualifications?"
    r"|projects?|achievements?|accomplishments?|awards?|languages?|references?|training"
    r"|publications?|volunteer\w*|interests?|hobbies)(?![a-z])[^\n]{0,40}$"
)
_PERSONAL_HEADER = re.compile(r"(?i)^[\W_]*personal\s*(?:informat\w*|details?|data|info)\b[\W_]*$")
#: Labels that may open a line of the personal block.
_PERSONAL_LABEL = re.compile(
    r"(?i)^[\W_]*(?:(?:full|first|last|middle|given|family|sur)\s*-?\s*name|name|e-?mail"
    r"|phone|mobile|tel(?:ephone)?|contact|address|location|city|linkedin|website|portfolio"
    r"|nationality|date of birth|dob|gender|age)\b"
)
_NAME_LABEL = re.compile(
    r"(?i)^[\W_]*(?:(?:full|first|last|middle|given|family|sur)\s*-?\s*name|name)\b"
)
#: A personal-block line longer than this, with no label and no "@", is content, not a name.
_MAX_NAME_LINE = 60
#: Capitalised words of a personal block, which a later line may repeat as a name.
_NAME_TOKEN = re.compile(r"\b[A-Z][a-zA-Z'\-]{2,}\b")
_NOT_A_NAME = frozenset((
    "Personal", "Information", "Informatics", "Details", "Name", "Names", "Full", "First",
    "Last", "Middle", "Given", "Family", "Surname", "Email", "Mail", "Phone", "Mobile",
    "Contact", "Address", "Location", "City", "Street", "Resume", "Same", "Candidate", "Data",
    "Masked", "Privacy", "Not", "Provided", "Linkedin", "Website", "Applicant", "Generated",
    "Experience", "Skills", "Education", "Summary", "Profile", "Objective", "Professional",
    "Career", "Curriculum", "Vitae",
))
#: A model meta-note opening a line: "Note:", "(Note: ...)", "**NOTE:**", "Please note that ...".
_NOTE_LINE = re.compile(
    r"(?i)^[\s*_#>\-(\[=~]*(?:(?:please|kindly)\s+note\b|(?:note|nb|n\.b\.|disclaimer|caveat)"
    r"\s*[:;\-–])"
)
#: A line of nothing but "#" and "*", e.g. the stray "#" that ends 4,506 RJDB CVs, or "**".
_HASH_LINE = re.compile(r"^[\s#*]*[#*][\s#*]*$")
#: A leaked "######" delimiter used as a heading: "###### Education", "#######Skills".
_DEEP_HEADING = re.compile(r"^(\s*)#{5,}\s*(?=\S)")
#: A run of three or more "#" glued after other text: "ronja.wilson#################".
_INLINE_HASHES = re.compile(r"(?<=[^\s#])#{3,}|\s+#{2,}\s*$")
#: The generator talking about its own construction, which survives as text the teacher would
#: read: "Same as Resume 1", "[[Keep the details exactly same ...]]", a mid-line "Note:".
#: A CV or JD with any of these left after cleaning is dropped, not repaired.
_META_RESIDUE = re.compile(
    r"(?i)\bnote\s*:"
    r"|\b(?:resume|cv)\s*#?\s*[12]\b"
    r"|\b(?:first|second|previous|original|initial|earlier|above|prior|matched|unmatched"
    r"|modified|altered) (?:resume|cv)\b"
    r"|\bsame as (?:before|above|resume|in resume|the (?:first|previous|original|above|matched))\b"
    r"|\[\[|\]\]|```"
    r"|\b(?:as )?per (?:the )?(?:task|instructions?|prompt)\b"
    r"|\bwithout including\b"
    r"|\b(?:skills?|experiences?|details?) (?:has|have|had) been (?:removed|excluded|omitted)\b"
    r"|\b(?:this|the) (?:resume|cv|profile|job description|information|details?|data|names?) "
    r"(?:is|are|was|were|has been|have been) (?:randomly )?(?:generated|simulated|fictional"
    r"|fictitious|hypothetical|made up)\b"
    r"|\b(?:simulated|dummy|fictional|fictitious|hypothetical|placeholder) (?:data|details"
    r"|information|names?|e-?mails?|purposes?|content)\b"
    r"|\billustrative purposes\b"
    r"|\bassum(?:e|ed|ing) (?:that )?the (?:applicant|candidate)\b"
    r"|\bas an ai (?:language )?model\b|\bI(?:'m| am) (?:sorry|unable)\b|\bI cannot\b"
)


def _clean_lines(lines: list[str], edits: dict | None = None) -> list[str]:
    """Drops meta-note and hash-only lines and normalises leaked hash runs. Nothing else.

    `edits`, when given, counts the texts each kind of edit touched, for the funnel.
    """
    out, notes, hashes = [], False, False
    for line in lines:
        if _NOTE_LINE.match(line):
            notes = True
            continue
        if _HASH_LINE.match(line):
            hashes = hashes or "#" in line
            continue
        fixed = _INLINE_HASHES.sub("", _DEEP_HEADING.sub(r"\1## ", line))
        hashes = hashes or fixed != line
        out.append(fixed)
    if edits is not None:
        edits["meta_note_lines_removed"] = edits.get("meta_note_lines_removed", 0) + notes
        edits["hash_artefacts_removed"] = edits.get("hash_artefacts_removed", 0) + hashes
    return out


def _join(lines: list[str]) -> str:
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def clean_rjdb_cv(text: str, known_name: str = "", edits: dict | None = None) -> tuple[str, str]:
    """(cleaned CV, "") or ("", the reason it cannot be cleaned safely).

    What is removed, and nothing else:

    * the personal block, i.e. everything before the first CV section. In RJDB that block
      holds the name and the e-mail. GPT-4 sampled the names to balance gender and race,
      so a name is an implicit gender and ethnicity cue that would contradict an injected
      gender, and both are PII. The block comes in every format: "## Personal
      Information", "**Personal Information**", bare name lines, a leading ":" or "**".
      A block that holds anything else, such as a line longer than a name with no label,
      is a reason to drop the CV, because then content would be cut;
    * model meta-notes ("Note: ...", "(Note: ...)", "Please note that ...");
    * leaked "#" delimiters: hash-only lines, "######" headings (normalised to "##"), and
      hash runs glued to text. A single "#" is kept, as in "C#".

    The CV is dropped, not repaired, if a name from the personal block (or `known_name`, the
    staged parse) is still in the body, or if construction meta-text is left over (see
    `_META_RESIDUE`). `edits`, when given, counts the CVs each kind of edit touched.
    """
    lines = str(text).replace("\r\n", "\n").split("\n")
    start = next((i for i, line in enumerate(lines) if _SECTION.match(line.strip())), None)
    if start is None:
        return "", "no_sections"

    names = set(_NAME_TOKEN.findall(str(known_name or "")))
    for line in lines[:start]:
        s = line.strip()
        if not s or _PERSONAL_HEADER.match(s) or re.fullmatch(r"[\W_]*", s):
            continue
        if "@" in s or _PERSONAL_LABEL.match(s):
            if _NAME_LABEL.match(s):
                names.update(_NAME_TOKEN.findall(s.split(":", 1)[-1] if ":" in s else s))
            continue
        if len(s) > _MAX_NAME_LINE:
            return "", "irregular_personal_block"
        names.update(_NAME_TOKEN.findall(s))

    if edits is not None and start > 0:
        edits["personal_block_removed"] = edits.get("personal_block_removed", 0) + 1
    body = _join(_clean_lines(lines[start:], edits))
    names -= _NOT_A_NAME
    if any(re.search(rf"\b{re.escape(n)}\b", body) for n in names):
        return "", "name_residue"
    if _META_RESIDUE.search(body):
        return "", "meta_text"
    return scrub_pii(body).strip(), ""


def clean_rjdb_jd(text: str, edits: dict | None = None) -> tuple[str, str]:
    """(cleaned JD, "") or ("", reason). The CV's artefact edits, without a personal block."""
    body = _join(_clean_lines(str(text).replace("\r\n", "\n").split("\n"), edits))
    if _META_RESIDUE.search(body):
        return "", "meta_text"
    return clean_jd(body), ""


# ---------------------------------------------------------------------------------------
# Kaggle category -> occupation, and the LinkedIn titles that belong to it
# ---------------------------------------------------------------------------------------

#: (include, exclude) title keywords for the 24 Kaggle categories. They are matched
#: case-insensitively against a Kaggle CV's headline (the LiveCareer LNAM field) and a
#: LinkedIn title, and a text belongs to a category when include matches and exclude does
#: not. The acquirer's TITLE_MAP and its spot checks were the starting point, with the
#: verifier's corrections applied. AUTOMOBILE now includes "automobile" and no longer
#: counts any "mechanic". "salesperson" is SALES. ADVOCATE means patient, customer and
#: victim advocates, not lawyers. Software engineers are IT, not ENGINEERING. ARTS
#: excludes "language arts" and "culinary arts".
CATEGORY_TITLE_KEYWORDS: dict[str, tuple[str, str | None]] = {
    "ACCOUNTANT": (
        r"\baccountants?\b|\baccounting\b|\bbookkeep|\bcpa\b|\bauditors?\b|\baudit\b"
        r"|accounts (?:payable|receivable)|\bpayroll\b|\btax (?:accountant|preparer|associate"
        r"|manager|analyst|specialist|senior|director|examin\w*)", None),
    "ADVOCATE": (r"\badvoca(?:te|tes|cy)\b", None),
    "AGRICULTURE": (
        r"\bagricultur\w*|\bagronom\w*|\bfarm(?:er|ers|ing|worker|hand)?\b|\blivestock\b"
        r"|\bhorticultur\w*|\bgreenhouse\b|\bcrops?\b|\bextension agent\b|\bagri\w*|\bgrain\b"
        r"|\bplant protection\b|\bforestry\b|\branch hand\b", r"\bserver farm\b|\bstate farm\b"),
    "APPAREL": (
        r"\bapparel\b|\bfashion\b|\bclothing\b|\bgarments?\b|\btextiles?\b|\bsoftlines\b"
        r"|\bknitwear\b|\bfootwear\b|\bwovens\b|\btailor\b", None),
    "ARTS": (
        r"\bartists?\b|\bart (?:director|teacher|instructor|educator|handler|therapist|manager)\b"
        r"|\barts\b|\bmuseum\b|\bgallery\b|\bcurator\b|\billustrator\b|\bmusician\b"
        r"|\btheat(?:er|re)\b|\bsculpt\w*|\bdancer\b",
        r"\blanguage arts\b|\bculinary arts\b|\bmartial arts\b|\bliberal arts\b"
        r"|\bmakeup artist\b|\bnail artist\b|\btattoo\b|\b(?:master|bachelor)s? of arts\b"),
    "AUTOMOBILE": (
        r"\bautomo(?:tive|bile)\b|\bauto (?:technician|tech|mechanic|body|parts|glass|detailer"
        r"|sales|service)\b|\bcar (?:wash|sales|dealer\w*)\b|\bdealership\b|\bcollision\b"
        r"|\bdiesel (?:mechanic|technician|tech)\b|\b(?:vehicle|fleet) (?:technician|mechanic"
        r"|inspector)\b|\bservice advisor\b|\btire (?:technician|tech)\b|\blube tech\b", None),
    "AVIATION": (
        r"\baviation\b|\baircraft\b|\bavionics\b|\bairlines?\b|\bairframe\b|\bramp agent\b"
        r"|\bflight (?:attendant|instructor|dispatcher|engineer|operations|crew|test)\b"
        r"|\bpilots?\b|\baerospace\b|\bair traffic\b|\ba&p\b|\bfaa\b", r"\bpilot plant\b"),
    "BANKING": (
        r"\bbank(?:ing|er|ers)?\b|\btellers?\b|\bloan (?:officer|processor|originator"
        r"|underwriter|specialist|servicing|closer)\b|\bmortgage\b|\bcredit (?:analyst|union)\b",
        r"\bfood bank\b|\bblood bank\b"),
    "BPO": (
        r"\bbpo\b|\bcall cent(?:er|re)\b|\bcontact cent(?:er|re)\b|\bcustomer (?:service|support"
        r"|care|services)\b|\btelesales\b|\btelemarket\w*|\bback office\b"
        r"|\bworkforce (?:manager|analyst|management)\b", None),
    "BUSINESS-DEVELOPMENT": (
        r"\bbusiness development\b|\bbdr\b|\bbd (?:manager|director|representative)\b"
        r"|\bpartnerships? (?:manager|director|lead)\b|\bbusiness developer\b", None),
    "CHEF": (
        r"\bchefs?\b|\bcooks?\b|\bculinary\b|\bkitchen\b|\bsous\b|\bpastry\b|\bbaker\b"
        r"|\bcatering\b", r"\bkitchen (?:designer|sales|cabinet\w*)\b"),
    "CONSTRUCTION": (
        r"\bconstruction\b|\bcarpent\w*|\belectricians?\b|\bplumb(?:er|ers|ing)\b|\bforem[ae]n\b"
        r"|\blaborers?\b|\bhvac\b|\bestimators?\b|\broofers?\b|\bmasons?\b|\bdrywall\b"
        r"|\bconcrete\b|\bpipefitters?\b|\bironworkers?\b|\bsite superintendent\b"
        r"|\bconstruction superintendent\b", None),
    "CONSULTANT": (r"\bconsultants?\b|\bconsulting\b", None),
    "DESIGNER": (r"\bdesigners?\b|\bux (?:research|researcher)\b|\bdrafters?\b|\bdraftsm[ae]n\b",
                 None),
    "DIGITAL-MEDIA": (
        r"\bdigital media\b|\bsocial media\b|\bdigital marketing\b|\bcontent (?:creator"
        r"|producer|strategist|manager|writer|specialist|marketing)\b|\bvideo (?:editor"
        r"|producer)\b|\bvideographer\b|\bmultimedia\b|\bseo\b|\bmedia (?:buyer|planner"
        r"|producer|specialist|coordinator)\b|\bdigital content\b|\bpodcast\w*", None),
    "ENGINEERING": (
        r"\bengineer(?:s|ing)?\b",
        r"\b(?:software|data|devops|devsecops|cloud|network|security|site reliability|desktop"
        r"|windows|linux|machine learning|ml"
        r"|ai|prompt|qa|test automation|full[- ]?stack|front[- ]?end|back[- ]?end|platform"
        r"|infrastructure|database|salesforce|java|python|\.net|sre|application|applications"
        r"|systems?|it|web|mobile|ios|android|integration|automation|sales|support|solutions"
        r"|pre-?sales|computer vision|nlp|big data|etl|bi|analytics|reliability) engineer"),
    "FINANCE": (
        r"\bfinanc\w*|\binvestments?\b|\btreasury\b|\bcontroller\b|\bfp&a\b|\bcfo\b"
        r"|\bwealth\b|\bportfolio manager\b",
        r"\b(?:air )?traffic controller\b|\binventory controller\b"),
    "FITNESS": (
        r"\bfitness\b|\bpersonal trainer\b|\byoga\b|\bpilates\b|\bgym\b"
        r"|\bstrength (?:and|&) conditioning\b|\bexercise\b|\bathletic trainer\b"
        r"|\bswim (?:instructor|coach)\b|\blifeguard\b",
        r"\bexercise (?:analyst|planner|designer|controller|officer)\b"),
    "HEALTHCARE": (
        r"\bnurses?\b|\bnursing\b|\brn\b|\blpn\b|\bcna\b|\bphysicians?\b|\bmedical\b"
        r"|\bclinical\b|\bclinic\b|\bhealth ?care\b|\bpharmac\w*|\btherapist\b|\bdental\b"
        r"|\bdentist\b|\bcaregivers?\b|\bpatients?\b|\bhospital\b|\bphlebotom\w*|\bradiolog\w*"
        r"|\bsonograph\w*|\bparamedic\b|\bemt\b|\bsurg(?:eon|ical)\b|\bhealth\b", None),
    "HR": (
        r"\bhr\b|\bhuman resources?\b|\brecruit(?:er|ers|ing|ment)?\b|\btalent acquisition\b"
        r"|\bpeople (?:operations|partner|ops)\b|\bhrbp\b|\bbenefits (?:specialist|administrator"
        r"|analyst|coordinator|manager)\b|\bpayroll\b|\bcompensation (?:analyst|manager)\b"
        r"|\bhris\b", r"(?:/|per )\s*hr\b"),
    "INFORMATION-TECHNOLOGY": (
        r"\bsoftware\b|\bdevelopers?\b|(?-i:\bIT\b)|\binformation technology\b"
        r"|\bsystems? administrator\b|\bsysadmin\b|\bnetwork (?:engineer|administrator|technician"
        r"|analyst|architect)\b|\bdevops\b|\bhelp ?desk\b|\bdesktop support\b|\bcyber ?security\b"
        r"|\binformation security\b|\bprogrammers?\b|\bdatabase\b|\bdba\b|\bcloud\b"
        r"|\bdata engineer\b|\bfull[- ]?stack\b|\bfront[- ]?end\b|\bback[- ]?end\b"
        r"|\bweb developer\b|\btechnical support\b|\bsolutions architect\b|\bsite reliability\b"
        r"|\bqa (?:engineer|analyst|tester)\b|\bsoftware tester\b|\betl\b|\bsap\b",
        r"\b(?:business|real estate|land|apparel|product|property|program|curriculum|fund"
        r"|resource|people|talent|leadership) developers?\b"),
    "PUBLIC-RELATIONS": (
        r"\bpublic relations\b|\bpr (?:manager|specialist|coordinator|director|associate"
        r"|account)\b|\bcommunications (?:manager|specialist|coordinator|director|associate"
        r"|officer|lead)\b|\bmedia relations\b|\bpublicist\b|\bpress (?:officer|secretary)\b"
        r"|\bcorporate communications\b", None),
    "SALES": (
        r"\bsales\b|\bsalesperson\b|\bsalesman\b|\baccount executive\b|\bretail associate\b",
        r"\bsales tax\b"),
    "TEACHER": (
        r"\bteach(?:er|ers|ing)\b|\btutors?\b|\binstructors?\b|\beducators?\b|\bprofessor\b"
        r"|\blecturer\b|\bparaprofessional\b|\bfaculty\b"
        r"|\b(?:school|assistant|vice|elementary|middle|high|deputy) principal\b", None),
}
_CATEGORY_RX = {
    cat: (re.compile(inc, re.I), re.compile(exc, re.I) if exc else None)
    for cat, (inc, exc) in CATEGORY_TITLE_KEYWORDS.items()
}


def title_categories(title: str) -> list[str]:
    """Every Kaggle category whose keywords a title or headline matches, in sorted order."""
    title = str(title or "")
    return [
        cat for cat, (inc, exc) in _CATEGORY_RX.items()
        if inc.search(title) and not (exc and exc.search(title))
    ]


def kaggle_occupation(category: str, headline: str) -> tuple[str, str]:
    """(occupation, rule) for a Kaggle CV; ("", reason) when there is no usable signal.

    `Category` is the LiveCareer search keyword the resume came back for, not an occupation.
    The verifier found headline/category agreement of 0-30% for AUTOMOBILE, AGRICULTURE,
    APPAREL, BPO, ARTS, AVIATION, FITNESS and BANKING. The headline is the CV's own job
    title, so it decides:

    * ``agree``: the headline matches the CV's own category -> that category;
    * ``headline``: the headline matches exactly one other category -> that one, because
      the CV says what it is, and the search keyword only says how it was found;
    * dropped: the headline is empty (``no_headline``), matches no category
      (``headline_unmapped``), or matches several categories, none of them its own
      (``headline_ambiguous``).
    """
    matched = title_categories(headline)
    if not str(headline or "").strip():
        return "", "no_headline"
    if category in matched:
        return category, "agree"
    if len(matched) == 1:
        return matched[0], "headline"
    return "", "headline_ambiguous" if matched else "headline_unmapped"


# ---------------------------------------------------------------------------------------
# Shared steps
# ---------------------------------------------------------------------------------------

#: One record of what every filter left: funnel[stage][step] = rows. Written to the dry-run
#: report and the generation report, because a yield is a finding about the source.
Funnel = dict


def _count(funnel: Funnel, stage: str, step: str, value) -> None:
    n = value if isinstance(value, (int, dict)) else len(value)
    funnel.setdefault(stage, {})[step] = n
    log.info("pool funnel %s/%s: %s", stage, step, n)


def source_rng(seed: int, name: str) -> np.random.Generator:
    """A generator per pool source, so adding one source never reshuffles another."""
    return np.random.default_rng([int(seed), zlib.crc32(name.encode())])


def drop_external_leaks(frame: pd.DataFrame, column: str, kind: str, root: Path,
                        funnel: Funnel, stage: str) -> pd.DataFrame:
    """Rows of `frame` whose `column` (a CV for kind="cv", a JD for "jd") copies no held-out text.

    The count of what was dropped goes into the funnel, keyed by the held-out source it
    copied. The 98 Kaggle CVs that FAIRE uses as bases land there as {"faire": 98}.
    """
    if frame.empty:
        return frame
    probe = frame[[column]].rename(columns={column: "_text"})
    cv_col, jd_col = ("_text", "_none") if kind == "cv" else ("_none", "_text")
    leaks = X.find_external_leaks(probe, cv_col=cv_col, jd_col=jd_col, root=root)
    by_source = (
        leaks["match"].str.split(":", n=1).str[0].value_counts().to_dict() if len(leaks) else {}
    )
    _count(funnel, stage, "holdout_leaks_by_source", {k: int(v) for k, v in by_source.items()})
    out = frame.drop(index=leaks["row"].unique()) if len(leaks) else frame
    if by_source:
        log.warning("%s: dropped %d row(s) that copy a held-out %s: %s",
                    stage, len(frame) - len(out), kind, by_source)
    return out


def _filter_cvs(frame: pd.DataFrame, cfg, funnel: Funnel, stage: str) -> pd.DataFrame:
    """The length bounds, then both marker lists, on the cleaned `cv` column."""
    lengths = frame["cv"].str.len()
    frame = frame[(lengths >= cfg.min_cv_chars) & (lengths <= cfg.max_cv_chars)]
    _count(funnel, stage, "length", frame)
    djinni_hit = frame["cv"].map(lambda t: PL.has_protected_marker(t, "en"))
    extra_hit = frame["cv"].map(has_external_extra_marker)
    _count(funnel, stage, "marker_hits_djinni_list", int(djinni_hit.sum()))
    _count(funnel, stage, "marker_hits_extra_list_only", int((extra_hit & ~djinni_hit).sum()))
    frame = frame[~(djinni_hit | extra_hit)]
    _count(funnel, stage, "no_protected_markers", frame)
    return frame


def _filter_jds(frame: pd.DataFrame, funnel: Funnel, stage: str) -> pd.DataFrame:
    """The Djinni JD length bounds, then the protected-requirement filter, on `job_description`."""
    lengths = frame["job_description"].str.len()
    frame = frame[(lengths >= PL.MIN_JOB_CHARS) & (lengths <= PL.MAX_JOB_CHARS)]
    _count(funnel, stage, "length", frame)
    return _drop_requirements(frame, funnel, stage)


def _drop_requirements(frame: pd.DataFrame, funnel: Funnel, stage: str) -> pd.DataFrame:
    reasons = frame["job_description"].map(jd_requirements)
    by_reason: dict[str, int] = {}
    for found in reasons:
        for name in found:
            by_reason[name] = by_reason.get(name, 0) + 1
    _count(funnel, stage, "requirement_hits_by_kind", dict(sorted(by_reason.items())))
    frame = frame[reasons.map(len) == 0]
    _count(funnel, stage, "no_protected_requirement", frame)
    return frame


def _reasons(values: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in values[values != ""].value_counts().items()}


# ---------------------------------------------------------------------------------------
# RJDB
# ---------------------------------------------------------------------------------------

def build_rjdb(cfg, n_pairs: int, rng: np.random.Generator, root: Path,
               funnel: Funnel) -> pd.DataFrame:
    """RJDB train triples -> pool rows: matched CV + JD, and optionally the weakened CV + JD.

    Only `split == "train"` is read, because dev and test are the held-out
    qualification-sensitivity set. A triple is eligible when its matched CV and its JD pass
    every filter. Triples are then taken in a seeded random order until `n_pairs` rows exist.
    With `cfg.rjdb_include_weakened`, a triple whose weakened CV also passes every filter
    contributes that CV with the same JD as a second row, `cv_variant="weakened"`. The two
    rows share `candidate_id` (`rjdb:<triple_id>`), so the split never separates a person
    from their own rewrite.
    """
    pairs = X.load_staged("rjdb", "pairs", root)
    cvs = X.load_staged("rjdb", "cvs", root).set_index("cv_id")
    jobs = X.load_staged("rjdb", "jobs", root).set_index("job_id")
    pairs = pairs[pairs["split"] == "train"]  # dev/test: the external holdout, never read
    matched = pairs[pairs["label"] == "match"].copy()
    _count(funnel, "rjdb_triples", "staged_train", matched)
    matched["triple_id"] = [X.parse_extra(e).get("triple_id", "") for e in matched["extra"]]
    if (matched["triple_id"] == "").any():
        raise ValueError("rjdb pairs.parquet: a train pair has no extra.triple_id")

    def clean_cvs(cv_ids: pd.Series, edits: dict) -> tuple[list[str], list[str]]:
        texts, reasons = [], []
        for cv_id in cv_ids:
            row = cvs.loc[cv_id]
            name = X.parse_extra(row["extra"]).get("name") or ""
            text, reason = clean_rjdb_cv(row["text"], name, edits)
            texts.append(text)
            reasons.append(reason)
        return texts, reasons

    frame = matched[["pair_id", "cv_id", "job_id", "triple_id"]].reset_index(drop=True)
    cv_edits: dict[str, int] = {}
    jd_edits: dict[str, int] = {}
    frame["cv"], cv_reasons = clean_cvs(frame["cv_id"], cv_edits)
    cleaned_jd = [clean_rjdb_jd(jobs.at[j, "text"], jd_edits) for j in frame["job_id"]]
    frame["job_description"] = [t for t, _ in cleaned_jd]
    jd_reasons = pd.Series([r for _, r in cleaned_jd], dtype=str)
    frame["job_position"] = [jobs.at[j, "title"] for j in frame["job_id"]]
    frame["occupation"] = [jobs.at[j, "occupation"] for j in frame["job_id"]]

    cv_reasons = pd.Series(cv_reasons, dtype=str)
    _count(funnel, "rjdb_cv", "staged_train_matched", frame)
    _count(funnel, "rjdb_cv", "cleaning_edits", cv_edits)
    _count(funnel, "rjdb_cv", "cleaning_drops_by_reason", _reasons(cv_reasons))
    _count(funnel, "rjdb_jd", "staged_train", frame)
    _count(funnel, "rjdb_jd", "cleaning_edits", jd_edits)
    _count(funnel, "rjdb_jd", "cleaning_drops_by_reason", _reasons(jd_reasons))
    frame = frame[(cv_reasons == "").to_numpy() & (jd_reasons == "").to_numpy()]
    _count(funnel, "rjdb_triples", "cv_and_jd_cleaned", frame)

    frame = _filter_cvs(frame, cfg, funnel, "rjdb_cv")
    frame = _filter_jds(frame, funnel, "rjdb_jd")
    frame = drop_external_leaks(frame, "cv", "cv", root, funnel, "rjdb_cv")
    frame = drop_external_leaks(frame, "job_description", "jd", root, funnel, "rjdb_jd")
    _count(funnel, "rjdb_triples", "eligible", frame)
    frame = frame.reset_index(drop=True)

    weakened: dict[str, tuple[str, str]] = {}
    if cfg.rjdb_include_weakened and not frame.empty:
        mismatch = pairs[pairs["label"] == "mismatch"]
        mismatch = mismatch.assign(
            triple_id=[X.parse_extra(e).get("triple_id", "") for e in mismatch["extra"]]
        ).set_index("triple_id")
        weak = mismatch.loc[frame["triple_id"], ["pair_id", "cv_id"]].reset_index()
        weak_edits: dict[str, int] = {}
        weak["cv"], weak_reasons = clean_cvs(weak["cv_id"], weak_edits)
        weak_reasons = pd.Series(weak_reasons, dtype=str)
        _count(funnel, "rjdb_weakened_cv", "of_eligible_triples", weak)
        _count(funnel, "rjdb_weakened_cv", "cleaning_edits", weak_edits)
        _count(funnel, "rjdb_weakened_cv", "cleaning_drops_by_reason", _reasons(weak_reasons))
        weak = weak[(weak_reasons == "").to_numpy()]
        weak = _filter_cvs(weak, cfg, funnel, "rjdb_weakened_cv")
        weak = drop_external_leaks(weak, "cv", "cv", root, funnel, "rjdb_weakened_cv")
        weakened = {t: (p, c) for t, p, c in zip(weak["triple_id"], weak["pair_id"], weak["cv"])}

    rows: list[dict] = []
    for position in rng.permutation(len(frame)):
        if len(rows) >= n_pairs:
            break
        triple = frame.iloc[int(position)]
        candidate_id = X.prefixed_id("rjdb", triple["triple_id"])
        job_id = X.prefixed_id("rjdb", triple["job_id"])
        variants = [(triple["pair_id"], triple["cv"], "matched")]
        if triple["triple_id"] in weakened:
            pair_id, cv = weakened[triple["triple_id"]]
            variants.append((pair_id, cv, "weakened"))
        for pair_id, cv, variant in variants:
            rows.append(_row(X.prefixed_id("rjdb", pair_id), candidate_id, job_id, cv,
                             triple["job_description"], triple["job_position"], "rjdb",
                             triple["occupation"], variant))
    out = pd.DataFrame(rows, columns=POOL_COLUMNS).head(n_pairs)
    _count(funnel, "rjdb_pairs", "sampled", out)
    _count(funnel, "rjdb_pairs", "by_cv_variant",
           {k: int(v) for k, v in out["cv_variant"].value_counts().items()})
    return out


# ---------------------------------------------------------------------------------------
# Kaggle resumes x LinkedIn postings
# ---------------------------------------------------------------------------------------

#: Two JDs this similar are one posting for pairing purposes. The overlap study's near rule
#: is Jaccard >= 0.7. LinkedIn has 30.8% removable near or exact duplicates, mostly
#: multi-location reposts and templated retail ads.
JD_NEAR_DUP_THRESHOLD = 0.7


class _NearDupSet:
    """JDs kept so far, by word 5-gram shingle. `add` refuses one too close to a kept JD."""

    def __init__(self, threshold: float = JD_NEAR_DUP_THRESHOLD):
        self.threshold = threshold
        self._docs: dict[int, list[int]] = {}
        self._sizes: list[int] = []

    def add(self, text: str) -> bool:
        hashes = X.shingle_hashes(text)
        if hashes.size == 0:
            return False
        shared: dict[int, int] = {}
        for h in hashes.tolist():
            for doc in self._docs.get(h, ()):
                shared[doc] = shared.get(doc, 0) + 1
        for doc, inter in shared.items():
            size = self._sizes[doc]
            score = max(inter / (hashes.size + size - inter), inter / hashes.size, inter / size)
            if score >= self.threshold:
                return False
        doc = len(self._sizes)
        self._sizes.append(int(hashes.size))
        for h in hashes.tolist():
            self._docs.setdefault(h, []).append(doc)
        return True


def _kaggle_candidates(cfg, root: Path, funnel: Funnel) -> pd.DataFrame:
    cvs = X.load_staged("kaggle_resume", "cvs", root)
    _count(funnel, "kaggle_cv", "staged", cvs)
    extra = cvs["extra"].map(X.parse_extra)
    frame = pd.DataFrame({
        "cv_id": cvs["cv_id"],
        "category": cvs["occupation"],
        "headline": [e.get("headline", "") for e in extra],
        "cluster": [e.get("near_dup_cluster", "") or "" for e in extra],
        "cv": cvs["text"].map(clean_kaggle_cv),
    })
    frame = _filter_cvs(frame, cfg, funnel, "kaggle_cv")
    # One CV per near-duplicate cluster. The dedup runs after the other filters, so a cluster
    # whose first member fails one of them is still represented by a member that passes.
    frame = frame.assign(_key=frame["cluster"].where(frame["cluster"] != "", frame["cv_id"]))
    frame = frame.drop_duplicates("_key", keep="first").drop(columns="_key")
    _count(funnel, "kaggle_cv", "near_duplicates_collapsed", frame)
    assigned = [kaggle_occupation(c, h) for c, h in zip(frame["category"], frame["headline"])]
    frame = frame.assign(occupation=[o for o, _ in assigned], rule=[r for _, r in assigned])
    _count(funnel, "kaggle_cv", "occupation_rule",
           {k: int(v) for k, v in frame["rule"].value_counts().items()})
    frame = frame[frame["occupation"] != ""]
    _count(funnel, "kaggle_cv", "occupation_assigned", frame)
    frame = drop_external_leaks(frame, "cv", "cv", root, funnel, "kaggle_cv")
    _count(funnel, "kaggle_cv", "eligible", frame)
    _count(funnel, "kaggle_cv", "eligible_by_occupation",
           {k: int(v) for k, v in frame["occupation"].value_counts().sort_index().items()})
    return frame.reset_index(drop=True)


def _linkedin_candidates(root: Path, funnel: Funnel) -> pd.DataFrame:
    jobs = X.load_staged("linkedin_postings", "jobs", root)
    _count(funnel, "linkedin_jd", "staged", jobs)
    jobs = jobs[jobs["lang"] == "en"]
    _count(funnel, "linkedin_jd", "english", jobs)
    frame = pd.DataFrame({
        "job_id": jobs["job_id"],
        "job_position": jobs["title"],
        "job_description": jobs["text"].map(clean_jd),
    })
    lengths = frame["job_description"].str.len()
    frame = frame[(lengths >= PL.MIN_JOB_CHARS) & (lengths <= PL.MAX_JOB_CHARS)]
    _count(funnel, "linkedin_jd", "length", frame)
    # Exact duplicates up to whitespace and case. Staged order decides which copy survives.
    key = frame["job_description"].map(lambda t: " ".join(t.casefold().split()))
    frame = frame[~key.duplicated(keep="first")]
    _count(funnel, "linkedin_jd", "exact_duplicates_collapsed", frame)
    # The title filter runs before the requirement filter only because it is cheap and halves
    # the rows the requirement regexes scan. The two commute.
    frame = frame.assign(categories=frame["job_position"].map(title_categories))
    frame = frame[frame["categories"].map(len) > 0]
    _count(funnel, "linkedin_jd", "title_in_a_category", frame)
    _count(funnel, "linkedin_jd", "title_in_several_categories",
           int((frame["categories"].map(len) > 1).sum()))
    frame = _drop_requirements(frame, funnel, "linkedin_jd")
    frame = drop_external_leaks(frame, "job_description", "jd", root, funnel, "linkedin_jd")
    _count(funnel, "linkedin_jd", "eligible", frame)
    return frame.reset_index(drop=True)


#: Title words that say nothing about the job itself, ignored when a CV headline is compared
#: with a JD title.
_TITLE_NOISE = frozenset((
    "senior", "sr", "junior", "jr", "lead", "staff", "i", "ii", "iii", "iv", "v", "part",
    "full", "time", "remote", "hybrid", "onsite", "contract", "temporary", "temp", "per",
    "diem", "prn", "level", "entry", "shift", "day", "night", "hour", "new", "job",
)) | X.STOPWORDS


def title_words(title: str) -> frozenset[str]:
    """The content words of a headline or title, which the headline-to-title ranking compares.

    Words split at every non-alphanumeric, so "HR ADMINISTRATOR/MARKETING" is three words.
    """
    words = re.findall(r"[a-z0-9]+", str(title or "").lower())
    return frozenset(w for w in words if w not in _TITLE_NOISE and len(w) > 1)


def build_kaggle_linkedin(cfg, n_pairs: int, rng: np.random.Generator, root: Path,
                          funnel: Funnel) -> pd.DataFrame:
    """Kaggle CVs, each paired with up to `cfg.jobs_per_candidate` LinkedIn JDs of its occupation.

    The match rule has two steps:

    1. *Occupation.* A JD fits a CV when the JD's title matches the CV's occupation in
       `CATEGORY_TITLE_KEYWORDS`. The occupation comes from `kaggle_occupation`, which is
       decided by the headline. A JD whose title matches several categories fits each of
       them.
    2. *Headline.* Among the fitting JDs, the ones whose titles share the most content words
       with the CV's headline (`title_words`) come first. A seeded shuffle breaks ties. A
       category is broad: TEACHER covers a preschool aide and a corrections instructor, and
       ADVOCATE covers a patient advocate and a cloud "ecosystem advocate". With the
       headline, a "PATIENT ADVOCATE" CV is offered "Patient Advocate" postings before any
       other advocate posting.

    The acquirer's precomputed `kaggle_resume_cats_by_*` fields are not used. They were hints
    for this map, and they are not ground truth.

    CVs are taken in a seeded random order. A JD is used at most once in the whole pool, and
    it is skipped when it is a near-duplicate of a JD already used (`JD_NEAR_DUP_THRESHOLD`).
    So no two pairs share a posting, nor a templated copy of one. A CV whose occupation has
    no JD left gets no pair. As in Djinni, these pairs are plausible, not ground truth.
    """
    cvs = _kaggle_candidates(cfg, root, funnel)
    jobs = _linkedin_candidates(root, funnel)

    # Per category: the JDs in a seeded random order, and an inverted index from each title
    # word to the positions (in that order) of the JDs whose title has it.
    order: dict[str, list[int]] = {}
    for index, categories in enumerate(jobs["categories"]):
        for category in categories:
            order.setdefault(category, []).append(index)
    order = {c: rng.permutation(np.asarray(order[c])).tolist() for c in sorted(order)}
    words = jobs["job_position"].map(title_words).tolist()
    by_word: dict[str, dict[str, list[int]]] = {}
    for category, indices in order.items():
        index_of = by_word.setdefault(category, {})
        for rank, index in enumerate(indices):
            for word in words[index]:
                index_of.setdefault(word, []).append(rank)

    kept = _NearDupSet()
    used: set[int] = set()
    rows: list[dict] = []
    shares_a_word: list[bool] = []
    near_dup_skipped = no_jobs = 0
    for position in rng.permutation(len(cvs)):
        if len(rows) >= n_pairs:
            break
        cv = cvs.iloc[int(position)]
        indices = order.get(cv["occupation"], [])
        overlap: dict[int, int] = {}
        for word in title_words(cv["headline"]):
            for rank in by_word.get(cv["occupation"], {}).get(word, ()):
                overlap[rank] = overlap.get(rank, 0) + 1
        ranked = itertools.chain(sorted(overlap, key=lambda r: (-overlap[r], r)),
                                 (r for r in range(len(indices)) if r not in overlap))
        picked: list[int] = []
        for rank in ranked:
            if len(picked) >= cfg.jobs_per_candidate:
                break
            index = indices[rank]
            if index in used:
                continue
            used.add(index)
            if not kept.add(jobs.at[index, "job_description"]):
                near_dup_skipped += 1
                continue
            picked.append(index)
            shares_a_word.append(overlap.get(rank, 0) > 0)
        if not picked:
            no_jobs += 1
            continue
        candidate_id = X.prefixed_id("kaggle_resume", cv["cv_id"])
        for index in picked:
            job = jobs.iloc[index]
            job_id = X.prefixed_id("linkedin_postings", job["job_id"])
            rows.append(_row(f"{candidate_id}__{job_id}", candidate_id, job_id, cv["cv"],
                             job["job_description"], job["job_position"], "kaggle_linkedin",
                             cv["occupation"], "original"))
    out = pd.DataFrame(rows, columns=POOL_COLUMNS).head(n_pairs)
    _count(funnel, "kaggle_linkedin_pairs", "jds_skipped_as_near_duplicates", near_dup_skipped)
    _count(funnel, "kaggle_linkedin_pairs", "pairs_sharing_a_headline_word",
           int(sum(shares_a_word[:len(out)])))
    _count(funnel, "kaggle_linkedin_pairs", "cvs_left_without_a_jd", no_jobs)
    _count(funnel, "kaggle_linkedin_pairs", "sampled", out)
    _count(funnel, "kaggle_linkedin_pairs", "distinct_cvs", int(out["candidate_id"].nunique()))
    _count(funnel, "kaggle_linkedin_pairs", "by_occupation",
           {k: int(v) for k, v in out["occupation"].value_counts().sort_index().items()})
    return out


# ---------------------------------------------------------------------------------------
# The external part of the pool
# ---------------------------------------------------------------------------------------

#: Every column `pool.match_pairs` emits, then the three a multi-source pool needs.
POOL_COLUMNS = [
    "pair_id", "candidate_id", "job_id", "cv", "job_description", "job_position",
    "primary_keyword", "candidate_years", "job_required_years", "lang",
    "source", "occupation", "cv_variant",
]


def _row(pair_id, candidate_id, job_id, cv, job_description, job_position, source,
         occupation, cv_variant) -> dict:
    """One external pool row. `primary_keyword` is Djinni's own taxonomy and is empty here.
    The two years columns are NaN because neither side states years (module docstring)."""
    return {
        "pair_id": pair_id, "candidate_id": candidate_id, "job_id": job_id, "cv": cv,
        "job_description": job_description, "job_position": str(job_position or ""),
        "primary_keyword": "", "candidate_years": float("nan"),
        "job_required_years": float("nan"), "lang": "en", "source": source,
        "occupation": str(occupation or ""), "cv_variant": cv_variant,
    }


BUILDERS = {"rjdb": build_rjdb, "kaggle_linkedin": build_kaggle_linkedin}


def external_sources(sources) -> list[str]:
    """The external names in `sources`, in order. An unknown name is an error, not a skip."""
    unknown = [s for s in sources if s not in POOL_SOURCES]
    if unknown:
        raise ValueError(f"unknown pool source(s) {unknown}; known: {sorted(POOL_SOURCES)}")
    return [s for s in sources if s != "djinni"]


def build_external(cfg, root: Path | None = None) -> tuple[pd.DataFrame, Funnel]:
    """Every external source in `cfg.sources`, concatenated in config order, plus the funnel.

    Each source gets its own seeded generator (`source_rng`). A source must have a count in
    `cfg.pairs_per_source`. A missing count is an error, because a default would silently
    decide how much teacher time a source gets.
    """
    root = Path(root) if root else X.external_root()
    names = external_sources(cfg.sources)
    funnel: Funnel = {}
    if not names:
        return pd.DataFrame(columns=POOL_COLUMNS), funnel
    if cfg.lang not in EXTERNAL_LANGS:
        log.warning("pool %s: external sources %s are English-only and are skipped",
                    cfg.lang, names)
        return pd.DataFrame(columns=POOL_COLUMNS), funnel
    missing = [s for s in names if s not in (cfg.pairs_per_source or {})]
    if missing:
        raise ValueError(f"pairs_per_source has no count for {missing}; set one per source")
    frames = []
    for name in names:
        part = BUILDERS[name](cfg, int(cfg.pairs_per_source[name]),
                              source_rng(cfg.seed, name), root, funnel)
        log.info("pool %s: %d pairs from %s (%d candidates, %d jobs)", cfg.lang, len(part),
                 name, part["candidate_id"].nunique(), part["job_id"].nunique())
        frames.append(part)
    return pd.concat(frames, ignore_index=True), funnel


# ---------------------------------------------------------------------------------------
# Publishing
# ---------------------------------------------------------------------------------------

def is_publishable(name: str) -> bool:
    """Whether rows of this pool source may be published.

    Djinni may be published. A pool source may be published only if every staged source it
    reads is publishable in `external.SOURCES`. A staged slug can also be named directly. An
    unknown name, including `MISSING_SOURCE`, may not.
    """
    if name == "djinni":
        return True
    staged = POOL_SOURCES.get(name, (name,))
    try:
        return bool(staged) and all(X.source(s).publishable for s in staged)
    except KeyError:
        return False


def unpublishable_rows(frame: pd.DataFrame) -> dict[str, int]:
    """{source: rows} for the rows of `frame` that may not be published.

    A frame without a `source` column predates multi-source pools and is Djinni. A null or
    empty value in a frame that has the column counts as `MISSING_SOURCE`.
    """
    if "source" not in frame.columns:
        return {}
    values = frame["source"].astype("object").where(frame["source"].notna(), MISSING_SOURCE)
    values = values.astype(str).replace("", MISSING_SOURCE)
    return {str(k): int(v) for k, v in values.value_counts().items() if not is_publishable(k)}


def check_publishable(frame: pd.DataFrame, where: str, allow: bool = False) -> dict[str, int]:
    """Refuses (SystemExit) to publish rows from unpublishable sources, unless `allow` is set.

    The publish scripts call it once per file. Every external training source is
    `publishable=False` until its licence and terms questions are settled. Several of them
    forbid redistribution, or are scrapes with unchecked terms, and publishing cannot be
    undone.
    """
    bad = unpublishable_rows(frame)
    if bad and not allow:
        raise SystemExit(
            f"{where}: refusing to publish {sum(bad.values()):,} row(s) from unpublishable "
            f"source(s) {bad}. Their licences are not settled (data/external.py SOURCES). Pass "
            "--allow-unpublishable-sources only after that has been decided."
        )
    if bad:
        log.warning("%s: publishing %s row(s) from unpublishable sources, as allowed", where, bad)
    return bad
