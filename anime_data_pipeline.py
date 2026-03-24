"""
╔══════════════════════════════════════════════════════════════════════════╗
║           ANIME DATASET — DATA SCIENCE PIPELINE                        ║
║  Raw Data → Cleaning → Missing Handling → Encoding → Scaling → Export  ║
╚══════════════════════════════════════════════════════════════════════════╝

Steps:
  1. Build raw data (as-is from source)
  2. Basic cleaning (duplicates, types, whitespace)
  3. Missing value handling (impute / flag)
  4. Feature engineering (season-level aggregates)
  5. Encoding (label + one-hot)
  6. Standardization & Scaling
  7. Validation
  8. Export clean CSV
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────
# STEP 0 ── RAW DATA DEFINITION
# ─────────────────────────────────────────────────────────────────────────

RAW_ANIME = [
    (1,  "Naruto",             "Action/Adventure","Pierrot",          3,  [(1,220,"8.4"),(2,500,"8.3"),(3,None,"8.2")],       "Naruto, Sasuke, Sakura, Kakashi",       2002,"Completed"),
    (2,  "One Piece",          "Action/Adventure","Toei Animation",   None,[(1,61,"8.9"),(2,77,"8.9"),(3,85,"8.8"),(4,None,"8.7")],"Luffy, Zoro, Nami, Usopp, Sanji",1999,"Ongoing"),
    (3,  "Attack on Titan",    "Action/Drama",    "MAPPA",            4,  [(1,25,"9.0"),(2,12,"9.1"),(3,22,"9.0"),(4,29,"9.1")], "Eren, Mikasa, Armin, Levi",           2013,"Completed"),
    (4,  "Death Note",         "Thriller/Mystery","Madhouse",         1,  [(1,37,"9.0")],                                       "Light, L, Ryuk, Misa",                2006,"Completed"),
    (5,  "FMA Brotherhood",    "Action/Fantasy",  "Bones",            1,  [(1,64,"9.1")],                                       "Edward, Alphonse, Mustang, Winry",    2009,"Completed"),
    (6,  "Dragon Ball Z",      "Action",          "Toei Animation",   None,[(1,39,"8.8"),(2,35,"8.7"),(3,None,"8.6"),(4,32,"8.8"),(5,26,"8.5")],"Goku, Vegeta, Gohan, Piccolo",1989,"Completed"),
    (7,  "Sword Art Online",   "Action/Fantasy",  "A-1 Pictures",     4,  [(1,25,"7.2"),(2,24,"7.0"),(3,24,"6.9"),(4,25,"7.4")], "Kirito, Asuna, Sinon, Alice",         2012,"Ongoing"),
    (8,  "My Hero Academia",   "Action/Superhero","Bones",            7,  [(1,13,"8.2"),(2,25,"8.4"),(3,25,"8.3"),(4,25,"8.5"),(5,25,"8.1"),(6,25,"8.3"),(7,None,"N/A")],"Izuku, Bakugo, Uraraka",2016,"Ongoing"),
    (9,  "Demon Slayer",       "Action/Fantasy",  "ufotable",         4,  [(1,26,"8.9"),(2,7,"9.1"),(3,11,"8.9"),(4,None,"9.0")], "Tanjiro, Nezuko, Zenitsu, Inosuke",  2019,"Ongoing"),
    (10, "Hunter x Hunter",    "Action/Adventure","Madhouse",         1,  [(1,148,"9.0")],                                      "Gon, Killua, Kurapika, Leorio",       2011,"Completed"),
    (11, "Tokyo Ghoul",        "Horror/Action",   "Pierrot",          4,  [(1,12,"7.8"),(2,12,"6.5"),(3,12,"7.3"),(4,12,"7.1")], "Kaneki, Touka, Hide, Amon",           2014,"Completed"),
    (12, "Bleach",             "Action/Adventure","Pierrot",          2,  [(1,366,"7.9"),(2,None,"8.2")],                        "Ichigo, Rukia, Orihime, Uryu",        2004,"Ongoing"),
    (13, "One Punch Man",      "Action/Comedy",   "Madhouse",         2,  [(1,12,"8.8"),(2,12,"7.4")],                          "Saitama, Genos, Bang, King",          2015,"Ongoing"),
    (14, "Fairy Tail",         "Action/Fantasy",  "A-1 Pictures",     3,  [(1,175,"8.0"),(2,102,"7.9"),(3,None,"7.8")],         "Natsu, Lucy, Gray, Erza",             2009,"Completed"),
    (15, "Steins;Gate",        "Sci-Fi/Thriller", "White Fox",        2,  [(1,24,"9.1"),(2,23,"8.2")],                          "Okabe, Kurisu, Mayuri, Daru",         2011,"Completed"),
    (16, "Code Geass",         "Mecha/Drama",     "Sunrise",          2,  [(1,25,"8.7"),(2,25,"8.9")],                          "Lelouch, Suzaku, C.C., Kallen",       2006,"Completed"),
    (17, "Neon Genesis Evangelion","Mecha/Psychological","Gainax",   1,  [(1,26,"8.5")],                                       "Shinji, Rei, Asuka, Misato",          1995,"Completed"),
    (18, "Cowboy Bebop",       "Sci-Fi/Noir",     "Sunrise",          1,  [(1,26,"8.9")],                                       "Spike, Jet, Faye, Edward",            1998,"Completed"),
    (19, "Vinland Saga",       "Historical/Action","MAPPA",           2,  [(1,24,"8.8"),(2,24,"9.0")],                          "Thorfinn, Askeladd, Canute, Einar",   2019,"Ongoing"),
    (20, "Jujutsu Kaisen",     "Action/Supernatural","MAPPA",         2,  [(1,24,"8.7"),(2,23,"9.0")],                          "Itadori, Gojo, Megumi, Nobara",       2020,"Ongoing"),
    (21, "Violet Evergarden",  "Drama/Fantasy",   "Kyoto Animation",  2,  [(1,13,"8.5"),(2,None,"8.9")],                        "Violet, Gilbert, Claudia, Cattleya",  2018,"Completed"),
    (22, "Your Lie in April",  "Drama/Music",     "A-1 Pictures",     1,  [(1,22,"8.7")],                                       "Kousei, Kaori, Tsubaki, Watari",      2014,"Completed"),
    (23, "Haikyuu!!",          "Sports/Drama",    "Production I.G",   4,  [(1,25,"8.7"),(2,25,"8.8"),(3,10,"8.7"),(4,25,"9.0")], "Hinata, Kageyama, Tsukishima, Daichi",2014,"Completed"),
    (24, "Re:Zero",            "Isekai/Fantasy",  "White Fox",        2,  [(1,25,"8.3"),(2,25,"8.6")],                          "Subaru, Emilia, Rem, Ram, Beatrice",  2016,"Ongoing"),
    (25, "Overlord",           "Isekai/Fantasy",  "Madhouse",         4,  [(1,13,"7.9"),(2,13,"7.8"),(3,13,"7.8"),(4,13,"7.8")], "Ainz, Albedo, Shalltear, Demiurge",  2015,"Ongoing"),
    (26, "Black Clover",       "Action/Fantasy",  "Pierrot",          1,  [(1,170,"7.7")],                                      "Asta, Yuno, Noelle, Yami",            2017,"Completed"),
    (27, "Mob Psycho 100",     "Action/Comedy",   "Bones",            3,  [(1,12,"8.5"),(2,13,"8.8"),(3,12,"8.8")],             "Mob, Reigen, Dimple, Ritsu",          2016,"Completed"),
    (28, "Assassination Classroom","Comedy/Action","Lerche",          2,  [(1,22,"8.1"),(2,25,"8.2")],                          "Nagisa, Karma, Koro-sensei, Kaede",   2015,"Completed"),
    (29, "Sword Art Online",   "Action/Fantasy",  "A-1 Pictures",     4,  [(1,25,"7.2"),(2,24,"7.0"),(3,24,"6.9"),(4,25,"7.4")], "Kirito, Asuna, Sinon, Alice",         2012,"Ongoing"),  # DUPLICATE
    (30, "No Game No Life",    "Isekai/Fantasy",  "Madhouse",         1,  [(1,12,"8.1")],                                       "Sora, Shiro, Stephanie, Jibril",      2014,"Completed"),
    (31, "Toradora!",          "Romance/Comedy",  "J.C.Staff",        1,  [(1,25,"8.2")],                                       "Ryuuji, Taiga, Minori, Yusaku",       2008,"Completed"),
    (32, "Clannad",            "Drama/Romance",   "Kyoto Animation",  2,  [(1,23,"8.8"),(2,24,"9.1")],                          "Tomoya, Nagisa, Kyou, Kotomi",        2007,"Completed"),
    (33, "Angel Beats!",       "Drama/Action",    "P.A.Works",        1,  [(1,13,"8.2")],                                       "Otonashi, Angel, Yuri, Hinata",       2010,"Completed"),
    (34, "Erased",             "Mystery/Thriller","A-1 Pictures",     1,  [(1,12,"8.6")],                                       "Satoru, Kayo, Airi, Kenya",           2016,"Completed"),
    (35, "Dr. Stone",          "Sci-Fi/Adventure","TMS Entertainment",3,  [(1,24,"8.3"),(2,11,"8.3"),(3,None,"8.5")],           "Senku, Taiju, Tsukasa, Chrome",       2019,"Ongoing"),
    (36, "The Promised Neverland","Thriller/Horror","CloverWorks",    2,  [(1,12,"8.6"),(2,11,"6.4")],                          "Emma, Ray, Norman, Isabella",         2019,"Completed"),
    (37, "Fruits Basket",      "Romance/Drama",   "TMS Entertainment",3,  [(1,13,"8.2"),(2,25,"8.7"),(3,13,"8.8")],            "Tohru, Kyo, Yuki, Shigure",           2019,"Completed"),
    (38, "FMA 2003",           "Action/Fantasy",  "Bones",            1,  [(1,51,"8.0")],                                       "Edward, Alphonse, Roy, Winry",        2003,"Completed"),
    (39, "Samurai Champloo",   "Action/Samurai",  "Manglobe",         1,  [(1,26,"8.5")],                                       "Mugen, Jin, Fuu",                     2004,"Completed"),
    (40, "Gurren Lagann",      "Mecha/Action",    "Gainax",           1,  [(1,27,"8.7")],                                       "Simon, Kamina, Yoko, Viral",          2007,"Completed"),
    (41, "Kill la Kill",       "Action/Comedy",   "Trigger",          1,  [(1,24,"8.1")],                                       "Ryuko, Satsuki, Mako, Senketsu",      2013,"Completed"),
    (42, "Noragami",           "Action/Fantasy",  "Bones",            2,  [(1,12,"8.1"),(2,13,"8.3")],                          "Yato, Hiyori, Yukine, Bishamonten",   2014,"Ongoing"),
    (43, "Soul Eater",         "Action/Fantasy",  "Bones",            1,  [(1,51,"7.9")],                                       "Maka, Soul, Black*Star, Death the Kid",2008,"Completed"),
    (44, "Blue Exorcist",      "Action/Fantasy",  "A-1 Pictures",     2,  [(1,25,"7.5"),(2,12,"7.7")],                          "Rin, Yukio, Shiemi, Ryuji",           2011,"Ongoing"),
    (45, "Darker than Black",  "Action/Sci-Fi",   "Bones",            2,  [(1,25,"8.1"),(2,12,"7.4")],                          "Hei, Yin, Misaki, Mao",               2007,"Completed"),
    (46, "Ouran Host Club",    "Romance/Comedy",  "Bones",            1,  [(1,26,"8.3")],                                       "Haruhi, Tamaki, Kyoya, Hikaru",       2006,"Completed"),
    (47, "Seraph of the End",  "Action/Supernatural","Wit Studio",    2,  [(1,12,"7.5"),(2,12,"7.6")],                          "Yuichiro, Mikaela, Shinoa, Guren",    2015,"Ongoing"),
    (48, "Shield Hero",        "Isekai",          "Kinema Citrus",    3,  [(1,25,"8.1"),(2,13,"7.6"),(3,12,"7.8")],             "Naofumi, Raphtalia, Filo, Itsuki",    2019,"Ongoing"),
    (49, "Mushishi",           "Fantasy/Mystery", "Artland",          2,  [(1,26,"8.7"),(2,20,"8.8")],                          "Ginko",                               2005,"Completed"),
    (50, "Made in Abyss",      "Adventure/Fantasy","Kinema Citrus",   2,  [(1,13,"8.7"),(2,12,"9.0")],                          "Riko, Reg, Nanachi, Prushka",         2017,"Ongoing"),
    (51, "Demon Slayer",       "Action/Fantasy",  "ufotable",         4,  [(1,26,"8.9"),(2,7,"9.1"),(3,11,"8.9"),(4,None,"9.0")], "Tanjiro, Nezuko, Zenitsu, Inosuke",  2019,"Ongoing"),  # DUPLICATE
    (52, "Danganronpa",        "Mystery/Thriller","Lerche",           3,  [(1,13,"7.5"),(2,13,"7.2"),(3,13,"7.4")],             "Naegi, Kirigiri, Togami, Junko",      2013,"Completed"),
    (53, "Accel World",        "Sci-Fi/Action",   "Sunrise",          1,  [(1,24,"7.4")],                                       "Haruyuki, Kuroyukihime, Takumu",      2012,"Completed"),
    (54, "Log Horizon",        "Isekai/Fantasy",  "Satelight",        3,  [(1,25,"7.9"),(2,25,"7.6"),(3,12,"7.5")],             "Shiroe, Akatsuki, Naotsugu, Nyanta",  2013,"Ongoing"),
    (55, "Madoka Magica",      "Drama/Magical",   "Shaft",            3,  [(1,12,"8.7"),(2,None,"8.5"),(3,None,"8.6")],         "Madoka, Homura, Sayaka, Mami",        2011,"Ongoing"),
    (56, "Psycho-Pass",        "Sci-Fi/Thriller", "Production I.G",   3,  [(1,22,"8.4"),(2,11,"7.8"),(3,8,"7.9")],              "Akane, Shinya, Makishima, Shougo",    2012,"Completed"),
    (57, "Space Brothers",     "Sci-Fi/Drama",    "A-1 Pictures",     1,  [(1,99,"8.7")],                                       "Mutta, Hibito, Serika, Sharon",       2012,"Completed"),
    (58, "Gintama",            "Comedy/Action",   "Bandai Namco",     5,  [(1,201,"8.9"),(2,51,"9.0"),(3,None,"9.1"),(4,13,"9.0"),(5,15,"9.1")],"Gintoki, Shinpachi, Kagura",2006,"Completed"),
    (59, "Nana",               "Drama/Music",     "Madhouse",         1,  [(1,47,"8.5")],                                       "Nana O., Nana K., Ren, Nobuo",        2006,"Completed"),
    (60, "Anohana",            "Drama",           "A-1 Pictures",     1,  [(1,11,"8.4")],                                       "Jinta, Menma, Yukiatsu, Tsuruko",     2011,"Completed"),
    (61, "Grave of the Fireflies","Drama/War",    "Studio Ghibli",    1,  [(1,1,"8.5")],                                        "Seita, Setsuko",                      1988,"Completed"),
    (62, "Princess Mononoke",  "Fantasy/Adventure","Studio Ghibli",   1,  [(1,1,"8.7")],                                        "Ashitaka, San, Lady Eboshi",          1997,"Completed"),
    (63, "Spirited Away",      "Fantasy",         "Studio Ghibli",    1,  [(1,1,"8.8")],                                        "Chihiro, Haku, Yubaba, Lin",          2001,"Completed"),
    (64, "Howl's Moving Castle","Fantasy/Romance","Studio Ghibli",    1,  [(1,1,"8.6")],                                        "Sophie, Howl, Calcifer, Markl",       2004,"Completed"),
    (65, "Your Name",          "Romance/Fantasy", "CoMix Wave",       1,  [(1,1,"8.9")],                                        "Taki, Mitsuha",                       2016,"Completed"),
    (66, "A Silent Voice",     "Drama/Romance",   "Kyoto Animation",  1,  [(1,1,"8.9")],                                        "Shoya, Shoko, Nagatsuka, Yuzuru",     2016,"Completed"),
    (67, "Weathering with You","Romance/Fantasy", "CoMix Wave",       1,  [(1,1,"8.5")],                                        "Hodaka, Hina, Nagi, Keisuke",         2019,"Completed"),
    (68, "Promare",            "Action/Mecha",    "Trigger",          1,  [(1,1,"8.0")],                                        "Galo, Lio, Kray, Aina",               2019,"Completed"),
    (69, "Berserk",            "Dark Fantasy",    "OLM",              3,  [(1,25,"8.7"),(2,24,"7.1"),(3,24,"6.9")],             "Guts, Griffith, Casca, Puck",         1997,"Ongoing"),
    (70, "Inuyasha",           "Action/Romance",  "Sunrise",          2,  [(1,167,"8.1"),(2,26,"8.0")],                         "Inuyasha, Kagome, Miroku, Sango",     2000,"Completed"),
    (71, "Sailor Moon",        "Magical/Romance", "Toei Animation",   5,  [(1,46,"8.1"),(2,43,"8.0"),(3,38,"8.1"),(4,39,"7.9"),(5,34,"7.9")],"Usagi, Ami, Rei, Makoto",1992,"Completed"),
    (72, "Cardcaptor Sakura",  "Magical/Romance", "Madhouse",         2,  [(1,70,"8.3"),(2,22,"8.6")],                          "Sakura, Syaoran, Tomoyo, Kero",       1998,"Completed"),
    (73, "Digimon Adventure",  "Adventure",       "Toei Animation",   3,  [(1,54,"8.0"),(2,50,"8.1"),(3,None,"7.6")],           "Tai, Matt, Sora, Izzy, Agumon",       1999,"Ongoing"),
    (74, "Pokemon",            "Adventure",       "OLM",              None,[(1,276,"7.5"),(2,192,"7.4"),(3,191,"7.3"),(4,193,"7.1")],"Ash, Pikachu, Misty, Brock",     1997,"Ongoing"),
    (75, "Dragon Ball",        "Action/Adventure","Toei Animation",   1,  [(1,153,"8.5")],                                      "Goku, Bulma, Krillin, Yamcha",        1986,"Completed"),
    (76, "Dragon Ball Super",  "Action",          "Toei Animation",   1,  [(1,131,"7.5")],                                      "Goku, Vegeta, Beerus, Whis",          2015,"Completed"),
    (77, "Yu-Gi-Oh!",          "Adventure",       "Gallop",           5,  [(1,27,"7.5"),(2,None,"7.6"),(3,None,"7.5"),(4,None,"7.4"),(5,None,"7.3")],"Yugi, Joey, Kaiba, Tea",2000,"Completed"),
    (78, "Beyblade",           "Sports/Action",   "Madhouse",         3,  [(1,51,"7.2"),(2,52,"7.1"),(3,52,"7.0")],             "Tyson, Kai, Ray, Max",                2001,"Completed"),
    (79, "Shaman King",        "Action/Supernatural","Xebec",         2,  [(1,64,"7.8"),(2,52,"7.7")],                          "Yoh, Anna, Horo Horo, Ren",           2001,"Ongoing"),
    (80, "Rurouni Kenshin",    "Action/Historical","Deen",            2,  [(1,95,"8.3"),(2,None,"8.7")],                        "Kenshin, Kaoru, Sanosuke, Yahiko",    1996,"Ongoing"),
    (81, "Black Butler",       "Mystery/Supernatural","A-1 Pictures", 3,  [(1,24,"7.8"),(2,12,"7.5"),(3,10,"7.6")],             "Ciel, Sebastian, Grell, Undertaker",  2008,"Ongoing"),
    (82, "Pandora Hearts",     "Fantasy/Mystery", "Xebec",            1,  [(1,25,"8.0")],                                       "Oz, Alice, Gilbert, Break",           2009,"Completed"),
    (83, "D.Gray-man",         "Action/Supernatural","TMS Entertainment",2,[(1,103,"8.0"),(2,13,"7.9")],                        "Allen, Lavi, Lenalee, Kanda",         2006,"Ongoing"),
    (84, "Blue Exorcist",      "Action/Fantasy",  "A-1 Pictures",     2,  [(1,25,"7.5"),(2,12,"7.7")],                          "Rin, Yukio, Shiemi, Ryuji",           2011,"Ongoing"),  # DUPLICATE title
    (85, "Magi",               "Action/Fantasy",  "A-1 Pictures",     2,  [(1,25,"8.1"),(2,25,"8.1")],                          "Aladdin, Alibaba, Morgiana, Sinbad",  2012,"Completed"),
    (86, "Akame ga Kill!",     "Action/Dark",     "White Fox",        1,  [(1,24,"7.8")],                                       "Tatsumi, Akame, Mine, Esdeath",       2014,"Completed"),
    (87, "Charlotte",          "Drama/Supernatural","P.A.Works",      1,  [(1,13,"7.9")],                                       "Yuu, Nao, Jojiro, Ayumi",             2015,"Completed"),
    (88, "Little Witch Academia","Fantasy/Comedy","Trigger",          1,  [(1,25,"8.0")],                                       "Akko, Lotte, Sucy, Ursula",           2017,"Completed"),
    (89, "Banana Fish",        "Drama/Thriller",  "MAPPA",            1,  [(1,24,"8.5")],                                       "Ash, Eiji, Shorter, Max",             2018,"Completed"),
    (90, "91 Days",            "Crime/Drama",     "Shuka",            1,  [(1,13,"8.1")],                                       "Avilio, Nero, Corteo, Fango",         2016,"Completed"),
    (91, "Dororo",             "Historical/Action","MAPPA",           1,  [(1,24,"8.3")],                                       "Hyakkimaru, Dororo, Tahomaru, Jukai", 2019,"Completed"),
    (92, "Megalo Box",         "Sports/Sci-Fi",   "TMS Entertainment",2,  [(1,13,"8.1"),(2,13,"8.0")],                          "Joe, Yuri, Nanbu, Shirato",           2018,"Completed"),
    (93, "Kabaneri",           "Action",          "Wit Studio",       1,  [(1,12,"7.5")],                                       "Ikoma, Mumei, Takumi, Ayame",         2016,"Completed"),
    (94, "Sirius the Jaeger",  "Action/Supernatural","P.A.Works",     1,  [(1,12,"7.7")],                                       "Yuliy, Philip, Mikhail, Willard",     2018,"Completed"),
    (95, "Given",              "Drama/Music",     "Lerche",           1,  [(1,11,"8.3")],                                       "Mafuyu, Ritsuka, Haruki, Akihiko",    2019,"Completed"),
    (96, "Sk8 the Infinity",   "Sports",          "Bones",            1,  [(1,12,"8.1")],                                       "Reki, Langa, Cherry, Joe",            2021,"Completed"),
    (97, "Carole & Tuesday",   "Music/Sci-Fi",    "Bones",            1,  [(1,24,"7.9")],                                       "Carole, Tuesday, Roddy, Angela",      2019,"Completed"),
    (98, "Link Click",         "Mystery/Drama",   "LAN Studio",       2,  [(1,11,"8.8"),(2,16,"9.0")],                          "Cheng Xiaoshi, Lu Guang, Qiao Ling",  2021,"Ongoing"),
    (99, "Chainsaw Man",       "Action/Horror",   "MAPPA",            1,  [(1,12,"8.6")],                                       "Denji, Power, Makima, Aki",           2022,"Ongoing"),
    (100,"Spy x Family",       "Comedy/Action",   "Wit/CloverWorks",  2,  [(1,25,"8.6"),(2,None,"8.6")],                        "Loid, Yor, Anya, Bond",               2022,"Ongoing"),
]


# ─────────────────────────────────────────────────────────────────────────
# STEP 1 ── LOAD RAW DATA INTO DATAFRAMES
# ─────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — LOADING RAW DATA")
print("=" * 65)

# ── Anime Master DataFrame ──
anime_records = []
for row in RAW_ANIME:
    aid, title, genre, studio, seasons, season_data, chars, year, status = row
    anime_records.append({
        "anime_id":       aid,
        "title":          title,
        "genre":          genre,
        "studio":         studio,
        "total_seasons":  seasons,   # intentionally has None
        "year_started":   year,
        "status":         status,
        "main_characters":chars,
    })
df_anime = pd.DataFrame(anime_records)

# ── Season-Level DataFrame ──
season_records = []
for row in RAW_ANIME:
    aid, title, genre, studio, seasons, season_data, chars, year, status = row
    for snum, eps, rating in season_data:
        season_records.append({
            "anime_id":     aid,
            "title":        title,
            "season_no":    snum,
            "episodes":     eps,      # some None
            "rating":       rating,   # some "N/A", some float strings
        })
df_seasons = pd.DataFrame(season_records)

print(f"  df_anime   : {df_anime.shape[0]} rows × {df_anime.shape[1]} cols")
print(f"  df_seasons : {df_seasons.shape[0]} rows × {df_seasons.shape[1]} cols")


# ─────────────────────────────────────────────────────────────────────────
# STEP 2 ── BASIC CLEANING
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 — BASIC CLEANING")
print("=" * 65)

# 2a. Strip whitespace from all string columns
for df in [df_anime, df_seasons]:
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).str.strip()

# 2b. Replace literal string "None" and "N/A" → np.nan
for df in [df_anime, df_seasons]:
    df.replace({"None": np.nan, "N/A": np.nan, "nan": np.nan}, inplace=True)

# 2c. Remove duplicate anime rows (same title + year)
before = len(df_anime)
df_anime.drop_duplicates(subset=["title", "year_started"], keep="first", inplace=True)
df_anime.reset_index(drop=True, inplace=True)
print(f"  Duplicates removed from df_anime : {before - len(df_anime)}")

# Also deduplicate seasons by anime_id + season_no (keep first)
before_s = len(df_seasons)
df_seasons.drop_duplicates(subset=["anime_id", "season_no"], keep="first", inplace=True)
df_seasons.reset_index(drop=True, inplace=True)
print(f"  Duplicate seasons removed        : {before_s - len(df_seasons)}")

# 2d. Cast types
df_anime["total_seasons"] = pd.to_numeric(df_anime["total_seasons"], errors="coerce")
df_anime["year_started"]  = pd.to_numeric(df_anime["year_started"],  errors="coerce").astype("Int64")
df_seasons["episodes"]    = pd.to_numeric(df_seasons["episodes"],    errors="coerce")
df_seasons["rating"]      = pd.to_numeric(df_seasons["rating"],      errors="coerce")

print(f"  df_anime after cleaning          : {df_anime.shape[0]} rows")
print(f"  df_seasons after cleaning        : {df_seasons.shape[0]} rows")


# ─────────────────────────────────────────────────────────────────────────
# STEP 3 ── MISSING VALUE HANDLING
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3 — MISSING VALUE HANDLING")
print("=" * 65)

print("\n  Missing values BEFORE imputation:")
print("  df_anime:\n",   df_anime.isnull().sum()[df_anime.isnull().sum() > 0].to_string())
print("\n  df_seasons:\n",df_seasons.isnull().sum()[df_seasons.isnull().sum() > 0].to_string())

# 3a. total_seasons → derive from season detail count where missing
season_counts = df_seasons.groupby("anime_id")["season_no"].max().reset_index()
season_counts.columns = ["anime_id", "derived_seasons"]
df_anime = df_anime.merge(season_counts, on="anime_id", how="left")
df_anime["total_seasons"] = df_anime["total_seasons"].fillna(df_anime["derived_seasons"])
df_anime.drop(columns=["derived_seasons"], inplace=True)
df_anime["total_seasons"] = df_anime["total_seasons"].astype(int)

# 3b. Season episodes → impute with median episodes per anime
ep_median = df_seasons.groupby("anime_id")["episodes"].transform("median")
df_seasons["episodes"] = df_seasons["episodes"].fillna(ep_median)
# Still any remaining → global median
global_ep_median = df_seasons["episodes"].median()
df_seasons["episodes"] = df_seasons["episodes"].fillna(global_ep_median)
df_seasons["episodes"] = df_seasons["episodes"].round().astype(int)
df_seasons["episode_imputed"] = df_seasons["episodes"].isnull()  # flag (already filled)

# 3c. Season rating → impute with mean rating of that anime's other seasons
rating_mean = df_seasons.groupby("anime_id")["rating"].transform("mean")
df_seasons["rating_imputed"] = df_seasons["rating"].isnull()
df_seasons["rating"] = df_seasons["rating"].fillna(rating_mean)
# Any still missing → global mean
df_seasons["rating"] = df_seasons["rating"].fillna(df_seasons["rating"].mean()).round(2)

print("\n  Missing values AFTER imputation:")
print("  df_anime:\n",   df_anime.isnull().sum()[df_anime.isnull().sum() > 0].to_string() or "   None!")
print("  df_seasons:\n", df_seasons.isnull().sum()[df_seasons.isnull().sum() > 0].to_string() or "   None!")


# ─────────────────────────────────────────────────────────────────────────
# STEP 4 ── FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4 — FEATURE ENGINEERING")
print("=" * 65)

# 4a. Aggregate season stats into anime-level
agg = df_seasons.groupby("anime_id").agg(
    total_episodes   = ("episodes", "sum"),
    avg_rating       = ("rating",   "mean"),
    max_rating       = ("rating",   "max"),
    min_rating       = ("rating",   "min"),
    rating_range     = ("rating",   lambda x: x.max() - x.min()),
).reset_index()
agg["avg_rating"] = agg["avg_rating"].round(2)
agg["rating_range"] = agg["rating_range"].round(2)

df_clean = df_anime.merge(agg, on="anime_id", how="left")

# 4b. Primary genre (before the /)
df_clean["primary_genre"] = df_clean["genre"].str.split("/").str[0].str.strip()

# 4c. Number of main characters
df_clean["char_count"] = df_clean["main_characters"].str.split(",").str.len()

# 4d. Anime age
current_year = 2024
df_clean["anime_age"] = current_year - df_clean["year_started"].astype(int)

# 4e. Is ongoing flag
df_clean["is_ongoing"] = (df_clean["status"] == "Ongoing").astype(int)

print(f"  Features after engineering : {df_clean.shape[1]} columns")
print(f"  Sample new columns: total_episodes, avg_rating, primary_genre, char_count, anime_age, is_ongoing")


# ─────────────────────────────────────────────────────────────────────────
# STEP 5 ── ENCODING
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5 — ENCODING")
print("=" * 65)

# 5a. Label Encode: status
le_status = LabelEncoder()
df_clean["status_encoded"] = le_status.fit_transform(df_clean["status"])
print(f"  Label Encoding 'status': {dict(zip(le_status.classes_, le_status.transform(le_status.classes_)))}")

# 5b. One-Hot Encode: primary_genre (top genres)
top_genres = df_clean["primary_genre"].value_counts().head(8).index.tolist()
for g in top_genres:
    df_clean[f"genre_{g.lower().replace(' ', '_').replace('/', '_')}"] = (df_clean["primary_genre"] == g).astype(int)
print(f"  One-Hot Encoding top 8 primary genres: {top_genres}")

# 5c. Label Encode: studio (for tree-based models)
le_studio = LabelEncoder()
df_clean["studio_encoded"] = le_studio.fit_transform(df_clean["studio"])
print(f"  Label Encoding 'studio': {df_clean['studio'].nunique()} unique studios → int codes")


# ─────────────────────────────────────────────────────────────────────────
# STEP 6 ── STANDARDIZATION & SCALING
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6 — STANDARDIZATION & SCALING")
print("=" * 65)

numeric_cols = ["total_seasons", "total_episodes", "avg_rating", "char_count", "anime_age", "rating_range"]

# 6a. StandardScaler (Z-score) — useful for linear models
scaler_std = StandardScaler()
std_scaled = scaler_std.fit_transform(df_clean[numeric_cols])
df_std = pd.DataFrame(std_scaled, columns=[f"{c}_zscore" for c in numeric_cols])

# 6b. MinMaxScaler (0–1) — useful for neural nets, distance-based
scaler_mm = MinMaxScaler()
mm_scaled = scaler_mm.fit_transform(df_clean[numeric_cols])
df_mm = pd.DataFrame(mm_scaled, columns=[f"{c}_minmax" for c in numeric_cols])

df_clean = pd.concat([df_clean.reset_index(drop=True), df_std, df_mm], axis=1)
print(f"  StandardScaler applied to  : {numeric_cols}")
print(f"  MinMaxScaler applied to    : {numeric_cols}")
print(f"  Total columns after scaling: {df_clean.shape[1]}")


# ─────────────────────────────────────────────────────────────────────────
# STEP 7 ── VALIDATION
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7 — VALIDATION")
print("=" * 65)

checks_passed = 0
checks_total  = 0

def check(label, condition):
    global checks_passed, checks_total
    checks_total += 1
    symbol = "✓" if condition else "✗"
    status = "PASS" if condition else "FAIL"
    print(f"  [{symbol}] {label:45s} → {status}")
    if condition:
        checks_passed += 1

check("No null values in df_clean",             df_clean.isnull().sum().sum() == 0)
check("anime_id is unique",                     df_clean["anime_id"].nunique() == len(df_clean))
check("total_seasons >= 1 for all",             (df_clean["total_seasons"] >= 1).all())
check("avg_rating in range 1–10",               df_clean["avg_rating"].between(1, 10).all())
check("year_started >= 1950",                   (df_clean["year_started"] >= 1950).all())
check("anime_age > 0",                          (df_clean["anime_age"] > 0).all())
check("total_episodes > 0",                     (df_clean["total_episodes"] > 0).all())
check("status_encoded has values only 0 or 1",  df_clean["status_encoded"].isin([0,1]).all())
check("MinMax rating_minmax in [0,1]",          df_clean["avg_rating_minmax"].between(0, 1).all())
check("No duplicate anime titles",              df_clean["title"].duplicated().sum() == 0)

print(f"\n  Validation: {checks_passed}/{checks_total} checks passed")


# ─────────────────────────────────────────────────────────────────────────
# STEP 8 ── EXPORT
# ─────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 8 — EXPORT")
print("=" * 65)

# Final clean master CSV
OUT_MASTER  = "anime_clean_master.csv"
OUT_SEASONS = "anime_clean_seasons.csv"
OUT_ML      = "anime_ml_ready.csv"

df_clean.to_csv(OUT_MASTER, index=False)
df_seasons.to_csv(OUT_SEASONS, index=False)

# ML-ready: only numeric + encoded columns
ml_cols = (
    ["anime_id", "total_seasons", "total_episodes", "avg_rating",
     "max_rating", "min_rating", "rating_range", "char_count",
     "anime_age", "is_ongoing", "status_encoded", "studio_encoded"]
    + [f"{c}_zscore" for c in numeric_cols]
    + [f"{c}_minmax" for c in numeric_cols]
    + [c for c in df_clean.columns if c.startswith("genre_")]
)
df_ml = df_clean[ml_cols]
df_ml.to_csv(OUT_ML, index=False)

print(f"  Exported: {OUT_MASTER}  ({df_clean.shape[0]} rows × {df_clean.shape[1]} cols)")
print(f"  Exported: {OUT_SEASONS} ({df_seasons.shape[0]} rows × {df_seasons.shape[1]} cols)")
print(f"  Exported: {OUT_ML}      ({df_ml.shape[0]} rows × {df_ml.shape[1]} cols — ML-ready)")

print("\n" + "=" * 65)
print("  PIPELINE COMPLETE!")
print("=" * 65)
print("""
  Files generated:
    anime_clean_master.csv   ← Full cleaned dataset (all features)
    anime_clean_seasons.csv  ← Season-level detail (cleaned)
    anime_ml_ready.csv       ← Numeric-only ML-ready dataset

  What to try next:
    • EDA  : df_clean['avg_rating'].hist()
    • ML   : predict avg_rating from total_episodes, seasons, genre
    • Viz  : seaborn.boxplot(x='primary_genre', y='avg_rating', data=df_clean)
    • NLP  : vectorize 'main_characters' with TF-IDF for similarity
""")
