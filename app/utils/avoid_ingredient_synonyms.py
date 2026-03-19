import re
import unicodedata
from typing import Dict, Iterable, List, Literal, Optional


AliasMode = Literal["input", "menu_strong", "menu_all"]


def normalize_ingredient_token(text: str) -> str:
    if not isinstance(text, str):
        return ""
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    normalized = "".join(
        ch for ch in unicodedata.normalize("NFKD", normalized) if not unicodedata.combining(ch)
    )
    return normalized


# catalog 구조 원칙:
# - display: 사용자에게 보여줄 표준명
# - input_aliases: 사용자가 자유형 입력으로 말할 수 있는 표현(비교적 넓게)
# - menu_evidence_aliases: 메뉴명에서 strong alias/direct 검증에 쓸 토큰
# - menu_prior_aliases: 메뉴명 자체가 특정 재료와 강하게 연결되는 대표 음식명
# - menu_weak_aliases: 메뉴명에서 약한 prior 정도로만 참고할 토큰
AVOID_INGREDIENT_CATALOG: Dict[str, Dict[str, object]] = {
    "egg": {
        "display": {"ko": "계란", "en": "egg", "cn": "鸡蛋"},
        "input_aliases": {
            "ko": ["계란", "달걀", "난류", "계란류", "달걀류", "계란흰자", "계란 노른자", "계란말이", "지단", "수란", "후라이"],
            "en": ["egg", "eggs", "egg white", "egg yolk", "omelet", "scrambled egg", "poached egg", "fried egg"],
            "cn": ["鸡蛋", "蛋", "蛋类", "蛋黄", "蛋白", "煎蛋", "蛋卷", "炒蛋"],
        },
        "menu_evidence_aliases": {
            "ko": ["계란", "달걀", "계란흰자", "계란 노른자"],
            "en": ["egg", "eggs", "egg white", "egg yolk"],
            "cn": ["鸡蛋", "蛋黄", "蛋白"],
        },
        "menu_weak_aliases": {
            "ko": ["계란말이", "지단", "수란", "후라이"],
            "en": ["omelet", "scrambled egg", "poached egg", "fried egg"],
            "cn": ["煎蛋", "蛋卷", "炒蛋"],
        },
        "menu_prior_aliases": {"ko": [], "en": [], "cn": []},
    },
    "dairy": {
        "display": {"ko": "유제품", "en": "dairy", "cn": "乳制品"},
        "input_aliases": {
            "ko": ["유제품", "우유류", "乳製品"],
            "en": ["dairy", "dairy product", "dairy products"],
            "cn": ["乳制品", "奶制品"],
        },
        "menu_evidence_aliases": {"ko": [], "en": [], "cn": []},
        "menu_prior_aliases": {"ko": [], "en": [], "cn": []},
        "menu_weak_aliases": {"ko": [], "en": [], "cn": []},
    },
    "milk": {
        "display": {"ko": "우유", "en": "milk", "cn": "牛奶"},
        "parents": ["dairy"],
        "input_aliases": {
            "ko": ["우유", "젖", "유당", "락토스", "연유", "분유"],
            "en": ["milk", "whole milk", "skim milk", "condensed milk", "lactose"],
            "cn": ["牛奶", "鲜奶", "炼乳", "乳糖"],
        },
        "menu_evidence_aliases": {
            "ko": ["우유", "연유", "분유"],
            "en": ["milk", "condensed milk"],
            "cn": ["牛奶", "炼乳"],
        },
        "menu_weak_aliases": {
            "ko": ["생크림"],
            "en": ["cream"],
            "cn": ["奶油"],
        },
        "menu_prior_aliases": {"ko": [], "en": [], "cn": []},
    },
    "cheese": {
        "display": {"ko": "치즈", "en": "cheese", "cn": "奶酪"},
        "parents": ["dairy"],
        "input_aliases": {
            "ko": ["치즈", "모짜렐라", "크림치즈", "파르메산", "체다치즈"],
            "en": ["cheese", "mozzarella", "cream cheese", "parmesan", "cheddar"],
            "cn": ["奶酪", "芝士", "马苏里拉", "奶油奶酪", "切达"],
        },
        "menu_evidence_aliases": {
            "ko": ["치즈", "모짜렐라", "크림치즈", "파르메산", "체다치즈"],
            "en": ["cheese", "mozzarella", "cream cheese", "parmesan", "cheddar"],
            "cn": ["奶酪", "芝士", "马苏里拉", "奶油奶酪", "切达"],
        },
        "menu_prior_aliases": {"ko": [], "en": [], "cn": []},
        "menu_weak_aliases": {"ko": [], "en": [], "cn": []},
    },
    "butter": {
        "display": {"ko": "버터", "en": "butter", "cn": "黄油"},
        "parents": ["dairy"],
        "input_aliases": {
            "ko": ["버터", "가염버터", "무염버터"],
            "en": ["butter", "salted butter", "unsalted butter"],
            "cn": ["黄油"],
        },
        "menu_evidence_aliases": {
            "ko": ["버터", "가염버터", "무염버터"],
            "en": ["butter", "salted butter", "unsalted butter"],
            "cn": ["黄油"],
        },
        "menu_prior_aliases": {"ko": [], "en": [], "cn": []},
        "menu_weak_aliases": {"ko": [], "en": [], "cn": []},
    },
    "peanut": {
        "display": {"ko": "땅콩", "en": "peanut", "cn": "花生"},
        "input_aliases": {
            "ko": ["땅콩", "땅콩류", "땅콩버터"],
            "en": ["peanut", "peanuts", "peanut butter"],
            "cn": ["花生", "花生酱", "花生碎"],
        },
        "menu_evidence_aliases": {
            "ko": ["땅콩", "땅콩버터"],
            "en": ["peanut", "peanuts", "peanut butter"],
            "cn": ["花生", "花生酱"],
        },
        "menu_prior_aliases": {"ko": [], "en": [], "cn": []},
        "menu_weak_aliases": {"ko": [], "en": [], "cn": []},
    },
    "tree nut": {
        "display": {"ko": "견과류", "en": "tree nut", "cn": "坚果"},
        "input_aliases": {
            "ko": ["견과류", "아몬드", "호두", "캐슈넛", "피스타치오", "헤이즐넛", "잣", "밤", "마카다미아"],
            "en": ["tree nut", "tree nuts", "almond", "walnut", "cashew", "pistachio", "hazelnut", "pine nut", "macadamia"],
            "cn": ["坚果", "杏仁", "核桃", "腰果", "开心果", "榛子", "松子", "栗子", "夏威夷果"],
        },
        "menu_evidence_aliases": {
            "ko": ["아몬드", "호두", "캐슈넛", "피스타치오", "헤이즐넛", "잣", "밤", "마카다미아"],
            "en": ["almond", "walnut", "cashew", "pistachio", "hazelnut", "pine nut", "macadamia"],
            "cn": ["杏仁", "核桃", "腰果", "开心果", "榛子", "松子", "栗子", "夏威夷果"],
        },
        "menu_weak_aliases": {"ko": [], "en": [], "cn": []},
    },
    "soy": {
        "display": {"ko": "대두", "en": "soy", "cn": "大豆"},
        "input_aliases": {
            "ko": ["대두", "콩", "콩류", "두부", "유부", "두유", "된장", "간장", "미소", "콩국", "콩국수"],
            "en": ["soy", "soybean", "soybeans", "tofu", "fried tofu", "soy milk", "miso", "soy sauce"],
            "cn": ["大豆", "黄豆", "豆类", "豆腐", "油豆腐", "豆浆", "味噌", "酱油"],
        },
        "menu_evidence_aliases": {
            "ko": ["대두", "콩", "두부", "유부", "콩국", "콩국수", "된장"],
            "en": ["soy", "soybean", "soybeans", "tofu", "fried tofu", "soy milk", "bean curd"],
            "cn": ["大豆", "黄豆", "豆腐", "油豆腐", "豆浆"],
        },
        "menu_weak_aliases": {
            "ko": ["간장", "미소"],
            "en": ["miso", "soy sauce"],
            "cn": ["味噌", "酱油"],
        },
        "menu_prior_aliases": {"ko": ["된장찌개"], "en": [], "cn": []},
    },
    "gluten": {
        "display": {"ko": "글루텐", "en": "gluten", "cn": "麸质"},
        "input_aliases": {
            "ko": ["글루텐"],
            "en": ["gluten"],
            "cn": ["麸质"],
        },
        "menu_evidence_aliases": {
            "ko": ["글루텐"],
            "en": ["gluten"],
            "cn": ["麸质"],
        },
        "menu_prior_aliases": {"ko": [], "en": [], "cn": []},
        "menu_weak_aliases": {"ko": [], "en": [], "cn": []},
    },
    "wheat": {
        "display": {"ko": "밀", "en": "wheat", "cn": "小麦"},
        "parents": ["gluten"],
        "input_aliases": {
            "ko": ["밀", "밀가루", "소맥", "글루텐", "면", "국수", "칼국수", "우동", "라면", "파스타", "수제비", "빵", "토스트", "만두피", "튀김옷"],
            "en": ["wheat", "flour", "gluten", "noodle", "noodles", "pasta", "udon", "ramen", "bread", "toast", "dumpling wrapper", "batter"],
            "cn": ["小麦", "面粉", "麸质", "面", "面条", "乌冬", "拉面", "面包", "饺子皮", "面糊"],
        },
        "menu_evidence_aliases": {
            "ko": ["밀", "밀가루", "소맥", "글루텐"],
            "en": ["wheat", "flour", "gluten", "wheat flour"],
            "cn": ["小麦", "面粉", "麸质"],
        },
        "menu_weak_aliases": {
            "ko": ["면", "국수", "칼국수", "우동", "라면", "파스타", "수제비", "빵", "토스트", "만두피", "튀김옷"],
            "en": ["noodle", "noodles", "pasta", "udon", "ramen", "bread", "toast", "dumpling wrapper", "batter"],
            "cn": ["面", "面条", "乌冬", "拉面", "面包", "饺子皮", "面糊"],
        },
        "menu_prior_aliases": {"ko": [], "en": [], "cn": []},
    },
    "shrimp": {
        "display": {"ko": "새우", "en": "shrimp", "cn": "虾"},
        "input_aliases": {
            "ko": ["새우", "새우살", "건새우", "새우젓", "크릴", "쉬림프"],
            "en": ["shrimp", "prawn", "dried shrimp", "shrimp paste", "shrimp ball", "krill"],
            "cn": ["虾", "虾仁", "虾皮", "虾酱", "虾丸", "明虾"],
        },
        "menu_evidence_aliases": {
            "ko": ["새우", "새우살", "건새우", "새우젓"],
            "en": ["shrimp", "prawn", "dried shrimp", "shrimp paste", "shrimp ball"],
            "cn": ["虾", "虾仁", "虾皮", "虾酱", "虾丸"],
        },
        "menu_prior_aliases": {"ko": [], "en": [], "cn": []},
        "menu_weak_aliases": {"ko": [], "en": [], "cn": []},
    },
    "crab": {
        "display": {"ko": "게", "en": "crab", "cn": "蟹"},
        "input_aliases": {
            "ko": ["게", "꽃게", "대게", "게살", "게장", "간장게장", "양념게장"],
            "en": ["crab", "crab meat", "blue crab", "snow crab", "soy-marinated crab"],
            "cn": ["蟹", "蟹肉", "螃蟹"],
        },
        "menu_evidence_aliases": {
            "ko": ["게", "꽃게", "대게", "게살", "게장", "간장게장", "양념게장"],
            "en": ["crab", "crab meat", "blue crab", "snow crab"],
            "cn": ["蟹", "蟹肉", "螃蟹"],
        },
        "menu_prior_aliases": {"ko": [], "en": [], "cn": []},
        "menu_weak_aliases": {"ko": [], "en": [], "cn": []},
    },
    "shellfish": {
        "display": {"ko": "패류", "en": "shellfish", "cn": "贝类"},
        "input_aliases": {
            "ko": ["패류", "조개", "조개류", "굴", "홍합", "전복", "가리비", "바지락", "꼬막", "소라", "모시조개", "키조개", "재첩"],
            "en": ["shellfish", "clam", "clams", "oyster", "mussel", "abalone", "scallop", "cockle", "whelk"],
            "cn": ["贝类", "蛤蜊", "牡蛎", "青口", "鲍鱼", "扇贝", "花蛤", "蛤", "海螺"],
        },
        "menu_evidence_aliases": {
            "ko": ["조개", "굴", "홍합", "전복", "가리비", "바지락", "꼬막", "소라", "모시조개", "키조개", "재첩"],
            "en": ["clam", "clams", "oyster", "mussel", "abalone", "scallop", "cockle", "whelk"],
            "cn": ["蛤蜊", "牡蛎", "青口", "鲍鱼", "扇贝", "花蛤", "蛤", "海螺"],
        },
        "menu_prior_aliases": {"ko": [], "en": [], "cn": []},
        "menu_weak_aliases": {"ko": [], "en": [], "cn": []},
    },
    "fish": {
        "display": {"ko": "생선", "en": "fish", "cn": "鱼"},
        "input_aliases": {
            "ko": ["생선", "어류", "멸치", "참치", "연어", "고등어", "명태", "동태", "북어", "황태", "코다리", "대구", "갈치", "삼치", "꽁치", "도미", "참도미", "광어", "우럭", "방어", "가자미", "도다리", "민어", "농어", "장어", "민물장어", "붕장어", "아나고", "복어", "메로", "아귀", "임연수", "전어", "청어", "병어", "가리비", "액젓", "생선육수", "회", "생선회", "어묵", "생선까스", "가쓰오", "가다랑어", "멸치육수"],
            "en": ["fish", "anchovy", "tuna", "salmon", "mackerel", "pollock", "cod", "dried pollock", "yellow pollock", "sea bream", "snapper", "halibut", "flounder", "rockfish", "yellowtail", "eel", "conger eel", "pufferfish", "anglerfish", "saury", "hairtail", "fish sauce", "fish stock", "sashimi", "fish cake", "fish cutlet", "bonito"],
            "cn": ["鱼", "鱼类", "凤尾鱼", "金枪鱼", "三文鱼", "鲭鱼", "明太鱼", "鳕鱼", "大头鱼", "真鲷", "比目鱼", "鲽鱼", "鲈鱼", "青花鱼", "带鱼", "秋刀鱼", "鳗鱼", "河鳗", "河豚", "安康鱼", "鱼露", "鱼汤", "刺身", "鱼饼"],
        },
        "menu_evidence_aliases": {
            "ko": ["생선", "어류", "멸치", "참치", "연어", "고등어", "명태", "동태", "북어", "황태", "코다리", "대구", "갈치", "삼치", "꽁치", "도미", "참도미", "광어", "우럭", "방어", "가자미", "도다리", "민어", "농어", "장어", "민물장어", "붕장어", "아나고", "복어", "메로", "아귀", "임연수", "전어", "청어", "병어", "회", "생선회", "어묵", "생선까스", "가쓰오", "가다랑어"],
            "en": ["fish", "anchovy", "tuna", "salmon", "mackerel", "pollock", "cod", "sea bream", "snapper", "halibut", "flounder", "rockfish", "yellowtail", "eel", "conger eel", "pufferfish", "anglerfish", "sashimi", "fish cake", "fish cutlet", "bonito"],
            "cn": ["鱼", "鱼类", "凤尾鱼", "金枪鱼", "三文鱼", "鲭鱼", "明太鱼", "鳕鱼", "真鲷", "比目鱼", "鲈鱼", "带鱼", "秋刀鱼", "鳗鱼", "河豚", "安康鱼", "刺身", "鱼饼"],
        },
        "menu_weak_aliases": {
            "ko": ["액젓", "생선육수", "멸치육수"],
            "en": ["fish sauce", "fish stock"],
            "cn": ["鱼露", "鱼汤"],
        },
        "menu_prior_aliases": {"ko": [], "en": [], "cn": []},
    },
    "sesame": {
        "display": {"ko": "참깨", "en": "sesame", "cn": "芝麻"},
        "input_aliases": {
            "ko": ["참깨", "참기름", "깨소금", "볶음참깨"],
            "en": ["sesame", "sesame seed", "sesame seeds", "sesame oil", "tahini"],
            "cn": ["芝麻", "芝麻酱", "芝麻油", "白芝麻", "黑芝麻"],
        },
        "menu_evidence_aliases": {
            "ko": ["참깨", "볶음참깨"],
            "en": ["sesame", "sesame seed", "sesame seeds"],
            "cn": ["芝麻", "白芝麻", "黑芝麻"],
        },
        "menu_weak_aliases": {
            "ko": ["참기름", "깨소금"],
            "en": ["sesame oil", "tahini"],
            "cn": ["芝麻酱", "芝麻油"],
        },
        "menu_prior_aliases": {"ko": [], "en": [], "cn": []},
    },
    "beef": {
        "display": {"ko": "소고기", "en": "beef", "cn": "牛肉"},
        "input_aliases": {
            "ko": ["소고기", "쇠고기", "우육", "한우", "육우", "소갈비", "우갈비", "갈비살", "차돌", "차돌박이", "양지", "사태", "우삼겹", "부채살", "살치살", "안창살", "토시살", "치마살", "업진살", "제비추리", "안심", "등심", "꽃등심", "채끝", "채끝살", "새우살", "우설", "우족", "우꼬리", "대창", "곱창", "염통", "막창", "토시스테이크", "갈비탕", "육회", "샤브샤브", "스키야키", "스테이크", "불고기", "대창전골", "곱창전골"],
            "en": ["beef", "beef brisket", "beef rib", "short rib", "brisket", "striploin", "sirloin", "tenderloin", "ribeye", "chuck flap", "flank steak", "skirt steak", "hanger steak", "tri-tip", "beef tripe", "beef large intestine", "beef omasum", "beef tongue", "beef tail", "beef broth", "steak", "sukiyaki", "shabu shabu", "bulgogi", "yukhoe"],
            "cn": ["牛肉", "牛", "牛腩", "牛肋排", "肥牛", "牛胸肉", "牛舌", "牛尾", "牛大肠", "牛小肠", "牛百叶", "牛心", "牛排", "寿喜烧", "涮锅", "烤肉", "牛汤"],
        },
        "menu_evidence_aliases": {
            "ko": ["소고기", "쇠고기", "우육", "한우", "육우", "소갈비", "우갈비", "갈비살", "차돌", "차돌박이", "양지", "사태", "우삼겹", "부채살", "살치살", "안창살", "토시살", "치마살", "업진살", "제비추리", "안심", "등심", "꽃등심", "채끝", "채끝살", "새우살", "우설", "우족", "우꼬리", "소막창", "소대창", "소곱창", "소염통"],
            "en": ["beef", "beef brisket", "short rib", "brisket", "striploin", "sirloin", "tenderloin", "ribeye", "flank steak", "skirt steak", "hanger steak", "tri-tip", "beef tongue", "beef tail", "beef tripe", "beef large intestine"],
            "cn": ["牛肉", "牛腩", "牛肋排", "肥牛", "牛胸肉", "牛舌", "牛尾", "牛百叶", "牛大肠"],
        },
        "menu_weak_aliases": {
            "ko": ["갈비탕", "불고기", "스테이크", "샤브샤브", "스키야키", "곱창", "대창", "막창", "염통"],
            "en": ["beef broth", "steak", "bulgogi", "shabu shabu", "sukiyaki", "tripe", "large intestine"],
            "cn": ["牛汤", "牛排", "寿喜烧", "涮锅"],
        },
        "menu_prior_aliases": {"ko": ["갈비탕", "토시스테이크", "대창전골", "곱창전골", "곱창볶음", "육회", "스키야키", "샤브샤브", "스테이크", "불고기"], "en": ["yukhoe", "sukiyaki", "shabu shabu", "bulgogi"], "cn": ["寿喜烧", "涮锅", "牛排"]},
    },
    "pork": {
        "display": {"ko": "돼지고기", "en": "pork", "cn": "猪肉"},
        "input_aliases": {
            "ko": ["돼지고기", "돈육", "돼지", "삼겹살", "삼겹", "오겹살", "목살", "항정살", "항정", "가브리살", "갈매기살", "앞다리살", "뒷다리살", "등갈비", "돼지갈비", "수육", "제육", "제육볶음", "두루치기", "돈까스", "돈카츠", "베이컨", "햄", "족발", "보쌈", "편육", "머리고기", "돼지껍데기", "껍데기", "꼬들살", "미추리", "오소리감투", "돼지막창", "돼지곱창", "막창", "곱창", "염통", "차슈", "돈코츠", "돼지국밥", "순대국"],
            "en": ["pork", "pig", "ham", "bacon", "pork belly", "pork loin", "pork shoulder", "jowl", "pork rib", "spare rib", "prosciutto", "lard", "char siu", "tonkotsu", "pork stock"],
            "cn": ["猪肉", "猪", "五花肉", "猪颈肉", "猪肩肉", "猪排", "猪肋排", "培根", "火腿", "叉烧", "猪骨汤", "猪大肠", "猪小肠", "猪肚"],
        },
        "menu_evidence_aliases": {
            "ko": ["돼지고기", "돈육", "돼지", "삼겹살", "삼겹", "오겹살", "목살", "항정살", "항정", "가브리살", "갈매기살", "앞다리살", "뒷다리살", "돼지갈비", "제육", "제육볶음", "돈까스", "돈카츠", "족발", "보쌈", "편육", "머리고기", "돼지껍데기", "꼬들살", "미추리", "오소리감투", "돼지막창", "돼지곱창", "돼지염통", "차슈", "돈코츠"],
            "en": ["pork", "pig", "pork belly", "pork loin", "pork shoulder", "jowl", "pork rib", "spare rib", "prosciutto", "char siu", "tonkotsu"],
            "cn": ["猪肉", "猪", "五花肉", "猪颈肉", "猪肩肉", "猪排", "猪肋排", "叉烧", "猪大肠", "猪小肠", "猪肚"],
        },
        "menu_weak_aliases": {
            "ko": ["수육", "베이컨", "햄", "껍데기", "막창", "곱창", "염통"],
            "en": ["ham", "bacon", "lard", "pork stock"],
            "cn": ["培根", "火腿", "猪骨汤"],
        },
        "menu_prior_aliases": {"ko": ["순대국", "돼지국밥", "보쌈정식", "제육볶음", "제육덮밥", "두루치기", "수육국밥", "차슈덮밥", "돈코츠라멘", "막창구이"], "en": ["char siu rice", "tonkotsu ramen"], "cn": ["叉烧饭", "猪骨拉面"]},
    },
    "chicken": {
        "display": {"ko": "닭고기", "en": "chicken", "cn": "鸡肉"},
        "input_aliases": {
            "ko": ["닭", "닭고기", "계육", "치킨", "닭다리", "닭다리살", "닭가슴살", "닭안심", "닭날개", "닭봉", "닭윙", "닭발", "닭목살", "닭근위", "닭염통", "모래집", "닭육수", "닭갈비", "닭볶음탕", "찜닭", "닭강정", "닭곰탕", "삼계탕", "백숙", "가라아게", "치킨가스", "치킨카츠", "치킨텐더", "치킨스테이크"],
            "en": ["chicken", "chicken thigh", "chicken breast", "chicken tender", "chicken wing", "drumstick", "gizzard", "chicken heart", "chicken stock", "poultry", "karaage", "fried chicken", "chicken katsu", "samgyetang"],
            "cn": ["鸡肉", "鸡", "鸡腿肉", "鸡胸肉", "鸡翅", "鸡腿", "鸡胗", "鸡心", "鸡汤", "炸鸡", "唐扬鸡", "参鸡汤"],
        },
        "menu_evidence_aliases": {
            "ko": ["닭", "닭고기", "계육", "치킨", "닭다리", "닭다리살", "닭가슴살", "닭안심", "닭날개", "닭봉", "닭윙", "닭발", "닭목살", "닭근위", "닭염통", "모래집"],
            "en": ["chicken", "chicken thigh", "chicken breast", "chicken tender", "chicken wing", "drumstick", "gizzard", "chicken heart", "fried chicken"],
            "cn": ["鸡肉", "鸡", "鸡腿肉", "鸡胸肉", "鸡翅", "鸡腿", "鸡胗", "鸡心"],
        },
        "menu_weak_aliases": {
            "ko": ["닭육수"],
            "en": ["chicken stock", "poultry"],
            "cn": ["鸡汤"],
        },
        "menu_prior_aliases": {"ko": ["삼계탕", "백숙", "찜닭", "닭갈비", "닭볶음탕", "닭강정", "닭곰탕", "가라아게", "치킨가스", "치킨카츠", "치킨텐더", "치킨스테이크"], "en": ["karaage", "samgyetang", "chicken katsu"], "cn": ["参鸡汤", "唐扬鸡", "炸鸡"]},
    },
    "lamb": {
        "display": {"ko": "양고기", "en": "lamb", "cn": "羊肉"},
        "input_aliases": {
            "ko": ["양고기", "양갈비", "램", "양육", "양꼬치", "양다리", "양사태", "양등심", "양어깨살", "프렌치랙", "램찹"],
            "en": ["lamb", "mutton", "lamb chop", "lamb rib", "lamb skewer", "leg of lamb", "lamb shank", "rack of lamb"],
            "cn": ["羊肉", "羊排", "羔羊肉", "羊肉串", "羊腿", "羊腱子肉"],
        },
        "menu_evidence_aliases": {
            "ko": ["양고기", "양갈비", "램", "양육", "양꼬치", "양다리", "양사태", "양등심", "양어깨살", "프렌치랙", "램찹"],
            "en": ["lamb", "mutton", "lamb chop", "lamb rib", "lamb skewer", "leg of lamb", "lamb shank", "rack of lamb"],
            "cn": ["羊肉", "羊排", "羔羊肉", "羊肉串", "羊腿", "羊腱子肉"],
        },
        "menu_weak_aliases": {"ko": [], "en": [], "cn": []},
        "menu_prior_aliases": {"ko": ["양꼬치", "램찹", "프렌치랙"], "en": ["lamb skewer", "rack of lamb"], "cn": ["羊肉串"]},
    },
    "duck": {
        "display": {"ko": "오리고기", "en": "duck", "cn": "鸭肉"},
        "input_aliases": {
            "ko": ["오리", "오리고기", "훈제오리", "오리훈제", "오리로스", "오리주물럭", "오리불고기", "오리백숙", "오리탕", "오리가슴살", "북경오리", "베이징덕"],
            "en": ["duck", "duck meat", "smoked duck", "duck breast", "roast duck", "peking duck"],
            "cn": ["鸭肉", "鸭", "熏鸭", "鸭胸", "烤鸭", "北京烤鸭"],
        },
        "menu_evidence_aliases": {
            "ko": ["오리", "오리고기", "훈제오리", "오리훈제", "오리로스", "오리주물럭", "오리불고기", "오리가슴살", "북경오리", "베이징덕"],
            "en": ["duck", "duck meat", "smoked duck", "duck breast", "roast duck", "peking duck"],
            "cn": ["鸭肉", "鸭", "熏鸭", "鸭胸", "烤鸭", "北京烤鸭"],
        },
        "menu_weak_aliases": {"ko": [], "en": [], "cn": []},
        "menu_prior_aliases": {"ko": ["오리탕", "오리백숙", "오리주물럭", "오리불고기", "북경오리", "베이징덕"], "en": ["peking duck"], "cn": ["北京烤鸭"]},
    },
}


def _iter_alias_terms(
    section: Literal["display", "input_aliases", "menu_evidence_aliases", "menu_weak_aliases"],
) -> Iterable[tuple[str, str]]:
    for canonical, meta in AVOID_INGREDIENT_CATALOG.items():
        values = meta.get(section, {})
        if not isinstance(values, dict):
            continue
        for localized_terms in values.values():
            if isinstance(localized_terms, str):
                yield canonical, localized_terms
            elif isinstance(localized_terms, list):
                for term in localized_terms:
                    if isinstance(term, str):
                        yield canonical, term


def _build_lookup(
    sections: List[Literal["display", "input_aliases", "menu_evidence_aliases", "menu_weak_aliases"]],
) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for section in sections:
        for canonical, term in _iter_alias_terms(section):
            normalized = normalize_ingredient_token(term)
            if normalized:
                lookup[normalized] = canonical

    for canonical in AVOID_INGREDIENT_CATALOG.keys():
        normalized_canonical = normalize_ingredient_token(canonical)
        if normalized_canonical:
            lookup[normalized_canonical] = canonical
    return lookup


INPUT_ALIAS_LOOKUP = _build_lookup(["display", "input_aliases"])
MENU_EVIDENCE_LOOKUP = _build_lookup(["display", "menu_evidence_aliases"])
MENU_ALL_ALIAS_LOOKUP = _build_lookup(["display", "menu_evidence_aliases", "menu_weak_aliases"])


def build_avoid_synonym_lookup() -> Dict[str, str]:
    # 하위 호환: 기존 함수명은 사용자 입력 정규화용 lookup을 반환한다.
    return dict(INPUT_ALIAS_LOOKUP)


def build_menu_evidence_lookup(include_weak: bool = False) -> Dict[str, str]:
    return dict(MENU_ALL_ALIAS_LOOKUP if include_weak else MENU_EVIDENCE_LOOKUP)


def get_menu_evidence_catalog() -> Dict[str, Dict[str, List[str]]]:
    catalog: Dict[str, Dict[str, List[str]]] = {}
    for canonical, meta in AVOID_INGREDIENT_CATALOG.items():
        display = meta.get("display", {})
        if not isinstance(display, dict):
            display = {}

        strong_terms: List[str] = []
        prior_terms: List[str] = []
        weak_terms: List[str] = []
        direct_terms: List[str] = []

        for value in display.values():
            if isinstance(value, str) and value.strip():
                direct_terms.append(value.strip())

        direct_terms.append(canonical)

        for section_name, bucket in (
            ("menu_evidence_aliases", strong_terms),
            ("menu_prior_aliases", prior_terms),
            ("menu_weak_aliases", weak_terms),
        ):
            values = meta.get(section_name, {})
            if not isinstance(values, dict):
                continue
            for localized_terms in values.values():
                if isinstance(localized_terms, str):
                    bucket.append(localized_terms)
                elif isinstance(localized_terms, list):
                    bucket.extend([term for term in localized_terms if isinstance(term, str)])

        def _dedupe(values: List[str]) -> List[str]:
            out: List[str] = []
            seen = set()
            for value in values:
                key = normalize_ingredient_token(value)
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(value)
            return out

        catalog[canonical] = {
            "direct": _dedupe(direct_terms),
            "strong": _dedupe(strong_terms),
            "prior": _dedupe(prior_terms),
            "weak": _dedupe(weak_terms),
        }
    return catalog


def get_canonical_ingredient(text: str, mode: AliasMode = "input") -> Optional[str]:
    normalized = normalize_ingredient_token(text)
    if not normalized:
        return None

    if mode == "input":
        return INPUT_ALIAS_LOOKUP.get(normalized)
    if mode == "menu_strong":
        return MENU_EVIDENCE_LOOKUP.get(normalized)
    return MENU_ALL_ALIAS_LOOKUP.get(normalized)


def get_canonical_parents(canonical: str) -> List[str]:
    meta = AVOID_INGREDIENT_CATALOG.get(canonical, {})
    parents = meta.get("parents", [])
    if not isinstance(parents, list):
        return []
    out: List[str] = []
    for parent in parents:
        if isinstance(parent, str) and parent.strip():
            out.append(parent.strip().casefold())
    return out


def get_canonical_ancestors(canonical: str) -> List[str]:
    canonical_norm = (canonical or "").strip().casefold()
    if not canonical_norm:
        return []

    ordered: List[str] = []
    seen = set()
    stack = list(get_canonical_parents(canonical_norm))
    while stack:
        current = stack.pop(0)
        if current in seen:
            continue
        seen.add(current)
        ordered.append(current)
        stack.extend(get_canonical_parents(current))
    return ordered


def find_matching_avoid_canonical(canonical: str, allowed_canonicals: Iterable[str]) -> Optional[str]:
    allowed = {
        str(value).strip().casefold()
        for value in allowed_canonicals
        if isinstance(value, str) and value.strip()
    }
    canonical_norm = (canonical or "").strip().casefold()
    if not canonical_norm:
        return None
    if canonical_norm in allowed:
        return canonical_norm
    for ancestor in get_canonical_ancestors(canonical_norm):
        if ancestor in allowed:
            return ancestor
    return None


def get_display_name(canonical: str, lang: str = "en") -> str:
    meta = AVOID_INGREDIENT_CATALOG.get(canonical, {})
    display = meta.get("display", {})
    if not isinstance(display, dict):
        return canonical

    if lang in display and isinstance(display[lang], str) and display[lang].strip():
        return display[lang].strip()

    for fallback in ("en", "ko", "cn"):
        value = display.get(fallback)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return canonical


def canonicalize_avoid_ingredients(ingredients: Iterable[str]) -> List[str]:
    canonicalized: List[str] = []
    seen = set()

    for ingredient in ingredients:
        normalized = normalize_ingredient_token(ingredient)
        if not normalized:
            continue
        canonical = INPUT_ALIAS_LOOKUP.get(normalized, re.sub(r"\s+", " ", str(ingredient)).strip())
        if canonical and canonical not in seen:
            seen.add(canonical)
            canonicalized.append(canonical)

    return canonicalized


def build_canonical_display_map(ingredients: Iterable[str], lang: str = "en") -> Dict[str, str]:
    display_map: Dict[str, str] = {}

    for ingredient in ingredients:
        normalized = normalize_ingredient_token(ingredient)
        if not normalized:
            continue

        raw_display = re.sub(r"\s+", " ", str(ingredient)).strip()
        canonical = INPUT_ALIAS_LOOKUP.get(normalized, raw_display).casefold()
        if canonical and canonical not in display_map:
            display_map[canonical] = raw_display

    for canonical in canonicalize_avoid_ingredients(ingredients):
        if canonical not in display_map:
            display_map[canonical] = get_display_name(canonical, lang=lang)

    return display_map


def build_avoid_lookup(ingredients: Iterable[str]) -> Dict[str, str]:
    avoid_lookup: Dict[str, str] = {}

    for ingredient in ingredients:
        normalized_original = normalize_ingredient_token(ingredient)
        if not normalized_original:
            continue

        display_name = re.sub(r"\s+", " ", str(ingredient)).strip()
        canonical = INPUT_ALIAS_LOOKUP.get(normalized_original, display_name)
        for token, mapped in INPUT_ALIAS_LOOKUP.items():
            if mapped == canonical:
                avoid_lookup[token] = display_name

        normalized_canonical = normalize_ingredient_token(canonical)
        if normalized_canonical:
            avoid_lookup[normalized_canonical] = display_name

        avoid_lookup[normalized_original] = display_name

    return avoid_lookup


def get_catalog_stats() -> Dict[str, int]:
    return {
        "canonical_count": len(AVOID_INGREDIENT_CATALOG),
        "input_alias_count": len(INPUT_ALIAS_LOOKUP),
        "menu_evidence_alias_count": len(MENU_EVIDENCE_LOOKUP),
        "menu_all_alias_count": len(MENU_ALL_ALIAS_LOOKUP),
    }


# 하위 호환: 예전 flat dict를 참조하는 코드는 input_aliases 기준으로만 보게 한다.
AVOID_INGREDIENT_SYNONYMS: Dict[str, List[str]] = {}
for _canonical, _meta in AVOID_INGREDIENT_CATALOG.items():
    _synonyms: List[str] = []
    _display = _meta.get("display", {})
    if isinstance(_display, dict):
        for _value in _display.values():
            if isinstance(_value, str):
                _synonyms.append(_value)
    _localized_synonyms = _meta.get("input_aliases", {})
    if isinstance(_localized_synonyms, dict):
        for _values in _localized_synonyms.values():
            if isinstance(_values, list):
                _synonyms.extend([_value for _value in _values if isinstance(_value, str)])
    deduped: List[str] = []
    seen = set()
    for _value in _synonyms:
        _key = normalize_ingredient_token(_value)
        if not _key or _key in seen:
            continue
        seen.add(_key)
        deduped.append(_value)
    AVOID_INGREDIENT_SYNONYMS[_canonical] = deduped
