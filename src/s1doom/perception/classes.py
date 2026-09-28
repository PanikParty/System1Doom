"""Actor class name -> normalized kind, plus per-monster danger/hp tables."""
from __future__ import annotations

KINDS = ("monster", "pickup_health", "pickup_armor", "pickup_ammo", "weapon", "key",
         "barrel", "projectile", "other")

# name: (hp, danger 0..1, short display name)
MONSTERS: dict[str, tuple[int, float, str]] = {
    "Zombieman": (20, 0.25, "zombieman"),
    "ShotgunGuy": (30, 0.45, "shotgunner"),
    "ChaingunGuy": (70, 0.6, "chaingunner"),
    "DoomImp": (60, 0.4, "imp"),
    "Demon": (150, 0.45, "demon"),
    "Spectre": (150, 0.5, "spectre"),
    "LostSoul": (100, 0.35, "lost_soul"),
    "Cacodemon": (400, 0.6, "cacodemon"),
    "HellKnight": (500, 0.65, "hell_knight"),
    "BaronOfHell": (1000, 0.75, "baron"),
    "Arachnotron": (500, 0.7, "arachnotron"),
    "PainElemental": (400, 0.55, "pain_elemental"),
    "Revenant": (300, 0.75, "revenant"),
    "Fatso": (600, 0.8, "mancubus"),
    "Archvile": (700, 0.95, "archvile"),
    "SpiderMastermind": (3000, 1.0, "spider_mastermind"),
    "Cyberdemon": (4000, 1.0, "cyberdemon"),
    "WolfensteinSS": (50, 0.4, "ss_guard"),
    # ViZDoom scenario actors
    "MarineChainsawVzd": (60, 0.5, "chainsaw_marine"),
}

PICKUPS: dict[str, str] = {
    **dict.fromkeys(["Stimpack", "Medikit", "HealthBonus", "Soulsphere", "Megasphere",
                     "Berserk", "CustomMedikit"], "pickup_health"),
    **dict.fromkeys(["ArmorBonus", "GreenArmor", "BlueArmor"], "pickup_armor"),
    **dict.fromkeys(["Clip", "ClipBox", "Shell", "ShellBox", "RocketAmmo", "RocketBox",
                     "Cell", "CellPack", "Backpack"], "pickup_ammo"),
    **dict.fromkeys(["Shotgun", "SuperShotgun", "Chaingun", "RocketLauncher", "PlasmaRifle",
                     "BFG9000", "Chainsaw"], "weapon"),
    **dict.fromkeys(["BlueCard", "RedCard", "YellowCard", "BlueSkull", "RedSkull",
                     "YellowSkull"], "key"),
    "ExplosiveBarrel": "barrel",
}

PROJECTILES = {"DoomImpBall", "CacodemonBall", "BaronBall", "Rocket", "ArachnotronPlasma",
               "RevenantTracer", "FatShot", "PlasmaBall", "BFGBall", "ArchvileFire"}

# Health restored by health pickups (for "worth_it")
HEALTH_VALUE = {"Stimpack": 10, "Medikit": 25, "CustomMedikit": 25, "HealthBonus": 1,
                "Soulsphere": 100, "Megasphere": 200, "Berserk": 100}
ARMOR_VALUE = {"ArmorBonus": 1, "GreenArmor": 100, "BlueArmor": 200}

# Items worth avoiding (ViZDoom health_gathering "Poison" vials)
HAZARDS = {"Poison"}

KEY_COLOR = {"BlueCard": "blue", "BlueSkull": "blue", "RedCard": "red", "RedSkull": "red",
             "YellowCard": "yellow", "YellowSkull": "yellow"}

# Doom weapon slot -> name (SELECTED_WEAPON values)
WEAPON_SLOTS = {1: "fist", 2: "pistol", 3: "shotgun", 4: "chaingun", 5: "rocket_launcher",
                6: "plasma_rifle", 7: "bfg"}
# weapon slot -> ammo game variable index
WEAPON_AMMO = {1: None, 2: 1, 3: 2, 4: 1, 5: 3, 6: 4, 7: 4}

IGNORED = {"DoomPlayer", "Blood", "BulletPuff", "TeleportFog", "ItemFog", "DeadMarine"}

# WAD thing type -> actor name (subset needed by the map model)
THING_TYPES = {
    5: "BlueCard", 6: "YellowCard", 13: "RedCard", 38: "RedSkull", 39: "YellowSkull",
    40: "BlueSkull", 2011: "Stimpack", 2012: "Medikit", 2014: "HealthBonus", 2015: "ArmorBonus",
    2018: "GreenArmor", 2019: "BlueArmor", 2013: "Soulsphere", 2007: "Clip", 2048: "ClipBox",
    2008: "Shell", 2049: "ShellBox", 2001: "Shotgun", 2002: "Chaingun", 2003: "RocketLauncher",
    2005: "Chainsaw", 2046: "RocketBox", 2010: "RocketAmmo", 2047: "Cell", 17: "CellPack",
    8: "Backpack", 82: "SuperShotgun", 2004: "PlasmaRifle", 2006: "BFG9000",
}


def kind_of(name: str, category: str | None = None) -> str:
    if name in MONSTERS:
        return "monster"
    if name in PICKUPS:
        return PICKUPS[name]
    if name in PROJECTILES:
        return "projectile"
    if name in IGNORED or name in HAZARDS:
        return "other"
    if category == "Monster":
        return "monster"
    return "other"


def monster_info(name: str) -> tuple[int, float, str]:
    return MONSTERS.get(name, (100, 0.5, name.lower()))
