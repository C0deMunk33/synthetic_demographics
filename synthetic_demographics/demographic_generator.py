import sqlite3
import pandas as pd
from pydantic import BaseModel
import random
import json
import time

from collections import Counter
from typing import List, Dict, Any, Optional

database_name = "synthetic_demographics/output_database.db"


class DemographicRecord(BaseModel):
    first_name: str
    last_name: str
    state: str
    age: int
    birthdate:str
    sex: str
    race: str
    is_student: bool
    education: str
    is_in_labor_force: bool
    is_employed: bool
    occupation_category: Optional[str]
    hobbies: List[str]
    aspirations: List[str]
    values: List[str]


class DemographicDatabase:
    def __init__(self, db_path: str = "output_database.db"):
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        
    def connect(self):
        """Establish database connection."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


class DemographicGenerator:
    def __init__(self, db_path: str = "output_database.db"):
        self.db_path = db_path
        self.db = DemographicDatabase(db_path)
        self.db.connect()

        # -------------
        # CACHES
        # -------------
        self.column_names_cache: List[str] = []
        
        # sc_est2023_alldata6 entire rows and probabilities (for picking dem rows)
        self.all_rows_cache: List[tuple] = []
        self.all_rows_probabilities_cache: List[float] = []
        
        # For state info
        self.state_abbr_cache: Dict[str, str] = {}  # full state name -> state abbr
        self.state_code_cache: Dict[str, int] = {}  # full state name -> numeric code

        # For population weight: (age, state_code) -> weight
        self.population_weights_cache: Dict[(int, int), float] = {}

        # For first-name picks: key = (sex_abbr, year, state_abbr_lower)
        # Value = List[(count, first_name, whi_prob, bla_prob, asi_prob, his_prob)] 
        #   or a fallback list if no results.
        self.first_name_cache: Dict[str, List] = {}

        # For last-name picks: key = "whi", "bla", "asi", "his"
        # Value = list of (last_name, race_prob)
        self.last_name_cache: Dict[str, List] = {}

        # For occupation+education: key = state_code
        #   "occupation_data": [ {occ_code, occ_title, weighted_count, education_props}, ... ]
        #   "employment_rate": float
        #   "labor_force_rate": float
        self.occupation_and_labor_cache: Dict[int, Dict[str, Any]] = {}

        # For birthdays
        # We'll store a structure: { month: {day: births_count}, ... }
        self.birthdays_cache: Dict[int, Dict[int, int]] = {}


        self.hobbies_cache: List[tuple] = []  # [(hobby, likelihood, skew)]
        self.aspirations_cache: List[tuple] = []  # [(aspiration, likelihood, skew)]
        self.values_cache: List[tuple] = []  # [(value, likelihood, skew)]

        # -------------------------------------------------------------------
        # Preload heavy data into memory for speed
        # -------------------------------------------------------------------
        self._cache_table_info()
        self._cache_all_rows()
        self._cache_birthdays()
        self._cache_state_info()  # read states for name→abbr mapping
        self._cache_state_codes() # read once from sc_est2023_alldata6 for name→state_code

        

    # --------------------------------------------------------------------------
    # GENERAL UTILITIES
    # --------------------------------------------------------------------------
    def _cache_table_info(self):
        """Cache the column names for sc_est2023_alldata6."""
        query = "PRAGMA table_info(sc_est2023_alldata6);"
        self.db.cursor.execute(query)
        columns = self.db.cursor.fetchall()
        self.column_names_cache = [col[1] for col in columns]

    def get_column_names(self):
        return self.column_names_cache

    # --------------------------------------------------------------------------
    # BIRTHDAYS
    # --------------------------------------------------------------------------
    def _cache_birthdays(self):
        """Cache all birthdays data (month, day, births)."""
        self.db.cursor.execute('''
            SELECT month, date_of_month, births
            FROM birthdays
        ''')
        rows = self.db.cursor.fetchall()
        for month, day, count in rows:
            if month not in self.birthdays_cache:
                self.birthdays_cache[month] = {}
            if day not in self.birthdays_cache[month]:
                self.birthdays_cache[month][day] = 0
            self.birthdays_cache[month][day] += count

    def get_birth_date(self):
        """
        Select birthday with weighting according to the births table,
        which tracks births by month/day.
        """
        months = list(self.birthdays_cache.keys())
        month_weights = [sum(self.birthdays_cache[m].values()) for m in months]
        selected_month = random.choices(months, weights=month_weights, k=1)[0]

        days_in_month = list(self.birthdays_cache[selected_month].keys())
        day_weights = list(self.birthdays_cache[selected_month].values())
        selected_day = random.choices(days_in_month, weights=day_weights, k=1)[0]

        return (selected_month, selected_day)

    # --------------------------------------------------------------------------
    # sc_est2023_alldata6: Weighted demographic picking
    # --------------------------------------------------------------------------
    def _cache_all_rows(self):
        """
        Fetch all rows (and their population weights) from sc_est2023_alldata6 
        with the typical filters used in _pick_demographic_rows, and store them 
        along with their probability distribution.
        """
        query = """
            SELECT *
            FROM sc_est2023_alldata6
            WHERE origin != 0
              AND sex != 0
              AND race NOT IN (3,5,6)
              AND age < 85 -- 85+ are lumped together
        """
        self.db.cursor.execute(query)
        rows = self.db.cursor.fetchall()
        if not rows:
            self.all_rows_cache = []
            self.all_rows_probabilities_cache = []
            return

        # The table schema we got from PRAGMA table_info. Suppose "popestimate2023" 
        # is at index 13 (this depends on actual DB structure; adjust as needed).
        # We'll find the correct index by name, to be robust:
        pop_col_idx = self.column_names_cache.index("popestimate2023")
        
        weights = [row[pop_col_idx] for row in rows]
        total_population = sum(weights)
        probabilities = [w / total_population for w in weights]

        self.all_rows_cache = rows
        self.all_rows_probabilities_cache = probabilities

    def _pick_demographic_rows(self, k=1):
        """
        Pick k demographic rows (weighted by population) from our cached data.
        """
        if not self.all_rows_cache or not self.all_rows_probabilities_cache:
            return []
        selected_rows = random.choices(
            self.all_rows_cache,
            weights=self.all_rows_probabilities_cache,
            k=k
        )
        return selected_rows

    # --------------------------------------------------------------------------
    # STATE HELPERS
    # --------------------------------------------------------------------------
    def _cache_state_info(self):
        """ Read the entire 'states' table: full state name (s.'_0') -> abbr (s.'_1'). """
        # We'll read all states from 'states' table into a dict
        query = 'SELECT s."_0", s."_1" FROM states s'
        self.db.cursor.execute(query)
        for full_name, abbr in self.db.cursor.fetchall():
            self.state_abbr_cache[full_name] = abbr

    def _cache_state_codes(self):
        """
        Retrieve a unique (name -> state_code) mapping from sc_est2023_alldata6,
        so we don't do repeated queries.
        """
        query = 'SELECT DISTINCT name, state FROM sc_est2023_alldata6'
        self.db.cursor.execute(query)
        results = self.db.cursor.fetchall()
        for state_name, code in results:
            self.state_code_cache[state_name] = code

    def _get_state_abbr(self, state_name: str) -> str:
        """Return the cached 2-letter state abbreviation for a full state name."""
        return self.state_abbr_cache.get(state_name, "")

    def _get_state_code(self, state_name: str) -> int:
        """Return the cached numeric state code for a full state name."""
        return self.state_code_cache.get(state_name, 0)

    # --------------------------------------------------------------------------
    # NAME HELPERS
    # --------------------------------------------------------------------------
    def _pick_first_name(self, sex, birth_year, race, origin, state_abbr):
        """
        Pick a first name from the cached DB data (weighted by popularity + race).
        We'll do exactly one query per (sex_abbr, year, state_abbr_lower) if not already cached,
        then at runtime combine with race/origin to form final race-based weighting.
        """
        sex_abbr = "F" if sex == 2 else "M"
        state_abbr_lower = state_abbr.lower()

        # We won't store separate caches for each race dimension. Instead, 
        # we'll store all 4 race probabilities for each name in the same list 
        # and do the weighting at runtime for each pick.
        cache_key = f"{sex_abbr}_{birth_year}_{state_abbr_lower}"

        # Race abbreviation for weighting
        # origin==2 => hispanic (?), race in {1=white,2=black,4=asian} 
        # We'll compute final prob = geometric mean( name_count_norm , race_prob ) for the appropriate race column
        # The appropriate race column:
        #   if origin=2 => use "his"
        #   elif race=2 => "bla"
        #   elif race=4 => "asi"
        #   else "whi"
        if origin == 2:
            race_col_idx = 5  # 'his' in the row
        elif race == 2:
            race_col_idx = 3  # 'bla'
        elif race == 4:
            race_col_idx = 4  # 'asi'
        else:
            race_col_idx = 2  # 'whi'

        if cache_key not in self.first_name_cache:
            # Attempt to query that state's sub-table for the given year & sex
            # For example: SELECT x."_4" as name_count, x."_3" as first_name, fn.whi, fn.bla, fn.asi, fn.his
            # from "ca" as x join first_racenameprobs fn on upper(x."_3")=fn.name ...
            query = f'''
                SELECT 
                    x."_4" as name_count, 
                    x."_3" as first_name,
                    fn.whi, fn.bla, fn.asi, fn.his
                FROM "{state_abbr_lower}" AS x
                JOIN first_racenameprobs fn 
                   ON (fn.name) = UPPER(x."_3")
                WHERE x."_1" = "{sex_abbr}"
                  AND x."_2" = {birth_year}
            '''
            self.db.cursor.execute(query)
            results = self.db.cursor.fetchall()

            if not results:
                # Fallback: if no data, load a small fallback from first_racenameprobs
                # This ensures we have at least some names
                self.db.cursor.execute("SELECT name, whi, bla, asi, his FROM first_racenameprobs LIMIT 200")
                fallback_results = []
                for (nm, w, b, a, h) in self.db.cursor.fetchall():
                    fallback_results.append((1, nm, w, b, a, h))
                self.first_name_cache[cache_key] = fallback_results
            else:
                self.first_name_cache[cache_key] = results

        # Now pick from the cached set
        name_list = self.first_name_cache[cache_key]
        if not name_list:
            # complete fallback
            return "John" if sex_abbr == "M" else "Mary"

        # We have rows like: (count, name, whi, bla, asi, his)
        # We'll compute combined weight = sqrt( (count_norm) * (race_prob) )
        max_count = max(row[0] for row in name_list)
        combined_weights = []
        names = []
        for row in name_list:
            count_val = row[0] / max_count if max_count else 0
            race_val = row[race_col_idx] if row[race_col_idx] else 0.0
            weight = (count_val * race_val) ** 0.5
            combined_weights.append(weight)
            names.append(row[1])  # the name

        total_w = sum(combined_weights)
        if total_w <= 0:
            return "John" if sex_abbr == "M" else "Mary"
        pick_probs = [w / total_w for w in combined_weights]
        selected_name = random.choices(names, weights=pick_probs, k=1)[0]
        return selected_name.title()

    def _pick_last_name(self, sex, birth_year, race, origin, state_abbr):
        """
        Pick a last name from the database, weighted by the chosen race distribution.
        We'll do a single query per 'race_abbr' if not cached, and store all rows in memory.
        """
        if origin == 2:
            race_abbr = "his"
        elif race == 2:
            race_abbr = "bla"
        elif race == 4:
            race_abbr = "asi"
        else:
            race_abbr = "whi"

        if race_abbr not in self.last_name_cache:
            query = f'''
                SELECT fn.name as last_name, fn.{race_abbr} as race_prob
                FROM last_racenameprobs fn
            '''
            self.db.cursor.execute(query)
            results = self.db.cursor.fetchall()
            if not results:
                # fallback
                self.last_name_cache[race_abbr] = [("Smith", 1.0)]
            else:
                self.last_name_cache[race_abbr] = results

        results = self.last_name_cache[race_abbr]
        weights = [float(rp) for _, rp in results]
        total_w = sum(weights)
        if total_w <= 0:
            return "Smith"

        pick_probs = [w / total_w for w in weights]
        names = [ln for ln, _ in results]
        try:
            return random.choices(names, weights=pick_probs, k=1)[0].title()
        except:
            return "Smith"

    # --------------------------------------------------------------------------
    # EDUCATION & OCCUPATION HELPERS
    # --------------------------------------------------------------------------
    def _get_possible_education_levels(self, age):
        """
        Returns possible education levels and approximate weights based on age.
        Also determines if the person is likely to be a student.
        """
        if age < 3:
            return {
                "levels": ["No High School"],
                "weights": [1.0],
                "is_student": False
            }
        
        elif age < 5:
            return {
                "levels": ["No High School"],
                "weights": [1.0],
                "is_student": (random.random() < 0.5)  # ~50% chance of preschool
            }
        
        elif age < 18:
            # K-12 range
            grade = age - 5  # approximate
            if grade < 9:  # Elementary/Middle
                return {
                    "levels": ["No High School"],
                    "weights": [1.0],
                    "is_student": True
                }
            else:  # High School
                dropout_rate = 0.05  # 5% assumption
                return {
                    "levels": ["No High School", "High School"],
                    "weights": [dropout_rate, 1 - dropout_rate],
                    "is_student": True
                }
        
        elif age < 22:
            # 18-21: Mix of HS, Some College, Associate's
            return {
                "levels": ["High School", "Some College", "Associate's Degree"],
                "weights": [0.3, 0.5, 0.2],
                "is_student": (random.random() < 0.6)
            }
        
        elif age < 25:
            # 22-24: Add Bachelor's possibility
            return {
                "levels": ["High School", "Some College", "Associate's Degree", "Bachelor's Degree"],
                "weights": [0.2, 0.3, 0.2, 0.3],
                "is_student": (random.random() < 0.3)
            }
        
        else:
            # 25+: All education levels possible, smaller student probability
            student_prob = max(0.01, min(0.2, 1 / (age - 20)))
            return {
                "levels": [
                    "No High School", "High School", "Some College", 
                    "Associate's Degree", "Bachelor's Degree", 
                    "Master's Degree", "Doctorate"
                ],
                "weights": None,  # We'll fill from occupation-based distribution
                "is_student": (random.random() < student_prob)
            }

    def _cache_occupation_and_labor_data(self, state_code: int):
        """
        Cache occupation data (plus labor force & employment rate) for a given state_code once.
        """
        if state_code in self.occupation_and_labor_cache:
            return  # Already cached

        # 1) Labor data
        labor_query = """
            SELECT 
                labor_force_pecent / 100.0 as labor_force_rate,
                CAST(REPLACE(employed_total, ',', '') AS FLOAT) /
                CAST(REPLACE(labor_force_total, ',', '') AS FLOAT) as employment_rate
            FROM labor_demographics 
            WHERE state_code = ? 
              AND pop_group = 'Total'
        """
        self.db.cursor.execute(labor_query, (state_code,))
        labor_result = self.db.cursor.fetchone()
        if labor_result and labor_result[0] and labor_result[1]:
            labor_force_rate, employment_rate = labor_result
        else:
            labor_force_rate, employment_rate = (0.6, 0.5)  # fallback

        # 2) Occupation & education data
        occupation_query = f"""
            SELECT 
                e.occ_code,
                e.occ_title,
                CAST(REPLACE(e.tot_emp, ',', '') AS FLOAT) as total_employment,
                ec.no_hs,
                ec.hs,
                ec.college_no_degree,
                ec.associates_degree,
                ec.bachelors_degree,
                ec.masters_degree,
                ec.doc_or_prof_degree
            FROM employment_data e
            JOIN education_by_career ec 
              ON e.occ_code = ec.occ_code
            WHERE e.area = {state_code}
              AND e.tot_emp IS NOT NULL
              AND e.tot_emp != '*'
              AND e.occ_code != '00-0000'
        """
        self.db.cursor.execute(occupation_query)
        occupation_data = self.db.cursor.fetchall()

        self.occupation_and_labor_cache[state_code] = {
            "labor_force_rate": labor_force_rate,
            "employment_rate": employment_rate,
            "occupation_data": occupation_data
        }

    def _get_population_weight_for_age_state(self, age: int, state_code: int) -> float:
        """
        Return the ratio of popestimate2023(age, state_code) to the sum of popestimate2023 for that
        age and state. We'll do this once per (age, state_code) or store in a cache.
        (In practice, sc_est2023_alldata6 has exactly one row per (age, state_code, race, sex, origin, etc.),
         so to do a single ratio: pop(age, state) / total_something. 
         If the table has multiple rows per (age, state), we sum them.)
        """
        key = (age, state_code)
        if key in self.population_weights_cache:
            return self.population_weights_cache[key]

        query = f"""
            SELECT 
                SUM(popestimate2023) 
            FROM sc_est2023_alldata6
            WHERE age = {age} 
              AND state = {state_code}
        """
        self.db.cursor.execute(query)
        sum_pop = self.db.cursor.fetchone()
        if not sum_pop or not sum_pop[0]:
            self.population_weights_cache[key] = 1.0
            return 1.0

        # For a simpler approach, we can just treat that sum as the "weight." 
        # Or interpret as fraction of the entire population. 
        # We'll store it as is (since the actual ratio might be 1 for everyone).
        self.population_weights_cache[key] = float(sum_pop[0])
        return float(sum_pop[0])

    def generate_education_and_occupation(self, age, state_code):
        """
        Generate a plausible education level and occupation (category/title) 
        given the person's age and state.
        """
        # 1) population weight for (age, state_code)
        pop_weight = self._get_population_weight_for_age_state(age, state_code)

        # 2) load occupation/labor data from cache
        self._cache_occupation_and_labor_data(state_code)
        cached_data = self.occupation_and_labor_cache.get(state_code, {})
        employment_rate = cached_data.get("employment_rate", 0.5)
        occupation_data = cached_data.get("occupation_data", [])

        # 3) Build weighted occupation data
        weighted_occupations = []
        for occ in occupation_data:
            # (occ_code, occ_title, total_employment, no_hs, hs, college_no_degree, 
            #  associates_degree, bachelors_degree, masters_degree, doc_or_prof_degree)
            occ_code, occ_title, total_emp = occ[0], occ[1], occ[2]
            edu_list = occ[3:]  # 7 items
            if not total_emp or total_emp < 0:
                continue

            # Weighted by pop_weight * employment_rate (rough approximation)
            w_count = total_emp * pop_weight * employment_rate
            if w_count <= 0:
                continue

            # compute edu distribution
            total_edu = sum(x for x in edu_list if x)
            if total_edu > 0:
                edu_props = [(x or 0) / total_edu for x in edu_list]
            else:
                edu_props = [0]*7

            weighted_occupations.append({
                'occ_code': occ_code,
                'occ_title': occ_title,
                'weighted_count': w_count,
                'education_props': edu_props
            })

        if not weighted_occupations:
            return {
                'age': age,
                'occupation_code': '00-0000',
                'occupation_title': 'Unknown Occupation',
                'education_level': 'High School',
                'is_student': False
            }

        total_weight = sum(o['weighted_count'] for o in weighted_occupations)
        pick_probs = [o['weighted_count']/total_weight for o in weighted_occupations]
        selected_occ = random.choices(weighted_occupations, weights=pick_probs, k=1)[0]

        # 4) Age-based possible education
        age_edu_info = self._get_possible_education_levels(age)
        possible_levels = age_edu_info["levels"]
        is_student = age_edu_info["is_student"]

        # 5) For adults, if we have weights=None, use occupation-based distribution
        if age_edu_info["weights"] is None:
            # match each of the 7 levels to an index in edu_props
            # (NoHS=0, HS=1, SomeCollege=2, Associate=3, Bachelor=4, Master=5, Doctorate=6)
            # We only select from the subset in 'possible_levels'
            all_levels = [
                "No High School", "High School", "Some College", 
                "Associate's Degree", "Bachelor's Degree", 
                "Master's Degree", "Doctorate"
            ]
            edu_probs = []
            for lvl in possible_levels:
                idx = all_levels.index(lvl)
                edu_probs.append(selected_occ['education_props'][idx])
        else:
            # use age-based distribution
            edu_probs = age_edu_info["weights"]

        # 6) pick final education
        s = sum(edu_probs)
        if s > 0:
            edu_probs = [p/s for p in edu_probs]
        else:
            edu_probs = [1/len(edu_probs)] * len(edu_probs)
        chosen_edu = random.choices(possible_levels, weights=edu_probs, k=1)[0]

        return {
            'age': age,
            'occupation_code': selected_occ['occ_code'],
            'occupation_title': selected_occ['occ_title'],
            'education_level': chosen_edu,
            'is_student': is_student
        }

    def generate_employment_status(self, age, state_code, education_level):
        """
        Generate whether person is in the labor force and if employed,
        using state-level labor rates (cached).
        """
        self._cache_occupation_and_labor_data(state_code)
        cached_data = self.occupation_and_labor_cache.get(state_code, {})
        labor_force_rate = cached_data.get("labor_force_rate", 0.6)
        employment_rate = cached_data.get("employment_rate", 0.5)

        # Is in labor force?
        is_in_labor_force = (random.random() <= labor_force_rate)

        if age < 16:
            return (False, False)
        
        if age < 18:
            is_in_labor_force = (random.random() <= 0.1)  # 10% chance for minors
        
        if not is_in_labor_force:
            return (False, False)

        # If in labor force, check employment
        is_employed = (random.random() <= employment_rate)
        return (True, is_employed)
    
    def _cache_hobbies(self):
        """Cache hobbies data if not already cached."""
        if not self.hobbies_cache:
            query = """
                SELECT hobby, adult_american_likelihood, male_female_skew
                FROM hobbylist_augmented
            """
            self.db.cursor.execute(query)
            results = self.db.cursor.fetchall()
            
            # Group by category
            for hobby, likelihood, skew in results:
                self.hobbies_cache.append((hobby, likelihood, skew))

    def _cache_aspirations(self):
        """Cache aspirations data if not already cached."""
        if not self.aspirations_cache:
            query = """
                SELECT aspiration_title, adult_american_likelihood, male_female_skew
                FROM aspirations_and_goals_augmented
                WHERE adult_american_likelihood > 0
            """
            self.db.cursor.execute(query)
            self.aspirations_cache = self.db.cursor.fetchall()

    def _cache_values(self):
        """Cache values data if not already cached."""
        if not self.values_cache:
            query = """
                SELECT value, adult_american_likelihood, male_female_skew
                FROM values_augmented
            """
            self.db.cursor.execute(query)
            self.values_cache = self.db.cursor.fetchall()

    def get_hobbies(self, age: int, sex: int) -> Dict[str, List[str]]:
        """
        Generate a list of current hobbies based on age. 0-2 hobbies per decade since 15.
        
        Args:
            age (int): Age of the person
            sex (int): Sex of the person (0 for male, 1 for female)
            
        Returns:
            Dict[str, List[str]]: Dictionary of hobbies organized by category
        """
        self._cache_hobbies()
        
        # Calculate number of potential hobby pickups
        # Start counting decades from age 15
        decades_since_15 = max(0, (age - 15) / 10)
        potential_hobby_count = round(decades_since_15 * 2)
        selected_hobby_count = random.randint(0, potential_hobby_count)
        if decades_since_15 > 1 and selected_hobby_count == 0:
            selected_hobby_count = 1
        
        # If no hobbies in cache, return empty result
        if not self.hobbies_cache:
            return {}
            
        # Calculate weights for each hobby based on likelihood and sex
        weights = []
        for hobby in self.hobbies_cache:
            hobby_name,adult_american_likelihood,male_female_skew = hobby
 
            base_weight = adult_american_likelihood
            # Adjust weight based on sex and male_female_skew
            # For males (sex=0), negative skew increases weight
            # For females (sex=1), positive skew increases weight
            gender_adjustment = male_female_skew * (2 * sex - 1)
            weight = base_weight * (1 + gender_adjustment)
            weights.append(max(0.1, weight))  # Ensure weight is at least 0.1
        
        # Select hobbies based on calculated weights
        selected_hobbies = []
        if selected_hobby_count > 0:
            selected_indices = random.choices(
                range(len(self.hobbies_cache)),
                weights=weights,
                k=min(selected_hobby_count, len(self.hobbies_cache))
            )
            selected_hobbies = [self.hobbies_cache[i] for i in selected_indices]
        
        # Organize selected hobbies by category
        hobbies = []
        for hobby in selected_hobbies:
            hobby_name,adult_american_likelihood,male_female_skew = hobby
            hobbies.append(hobby_name)
        
        return hobbies
        
    def get_aspirations(self, age: int, sex: int) -> List[str]:
        """
        Generate a list of aspirations based on age.
        
        Args:
            age (int): Person's age
            
        Returns:
            List[str]: List of aspirations
        """
        self._cache_aspirations()
        
        # Similar to hobbies, calculate potential aspirations
        # Start counting decades from age 15
        decades_since_15 = max(0, (age - 15) / 10)
        potential_aspiration_count = round(decades_since_15 * 2)
        selected_aspiration_count = random.randint(0, potential_aspiration_count)
        if decades_since_15 > 1 and selected_aspiration_count == 0:
            selected_aspiration_count = 1

        if not self.aspirations_cache:
            return []
        
        # Calculate weights for each aspiration based on likelihood
        # fields: aspiration_title,adult_american_likelihood,male_female_skew
        weights = []
        for aspiration in self.aspirations_cache:
            aspiration_title, likelihood, skew = aspiration
            base_weight = likelihood
            skew_adjustment = skew * (2 * sex - 1)
            weight = base_weight * (1 + skew_adjustment)
            weights.append(max(0.1, weight))

        # Select aspirations based on calculated weights
        selected_aspirations = []
        if selected_aspiration_count > 0:
            selected_indices = random.choices(
                range(len(self.aspirations_cache)),
                weights=weights,
                k=min(selected_aspiration_count, len(self.aspirations_cache))
            )
            selected_aspirations = [self.aspirations_cache[i] for i in selected_indices]

        return [a[0] for a in selected_aspirations]

    def get_values(self, age: int, sex: int) -> List[str]:
        """
        Generate a list of personal values.
        
        Args:
            age (int): Person's age
            
        Returns:
            List[str]: List of values
        """
        self._cache_values()
        
        # Calculate potential value count based on age
        # Start counting decades from age 15
        decades_since_15 = max(0, (age - 15) / 10)
        potential_value_count = round(decades_since_15 * 2)
        selected_value_count = random.randint(0, potential_value_count)
        if decades_since_15 > 1 and selected_value_count == 0:
            selected_value_count = 1

        if not self.values_cache:
            return []
        
        # Calculate weights for each value based on likelihood
        weights = []
        for value in self.values_cache:
            value_name, likelihood, skew = value
            base_weight = likelihood
            skew_adjust = skew * (2 * sex - 1)
            weight = base_weight * (1 + skew_adjust)
            weights.append(max(0.1, weight))

        # Select values based on calculated weights
        selected_values = []
        if selected_value_count > 0:
            selected_indices = random.choices(
                range(len(self.values_cache)),
                weights=weights,
                k=min(selected_value_count, len(self.values_cache))
            )
            selected_values = [self.values_cache[i] for i in selected_indices]

        return [v[0] for v in selected_values]
            

    # --------------------------------------------------------------------------
    # BATCH GENERATION
    # --------------------------------------------------------------------------
    def generate_demographic_batch(self, samples=100) -> List[Dict[str, Any]]:
        """
        Generate a batch of demographics in one shot.
        
        Args:
            samples (int): Number of demographic records to produce.
            
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing 
                                  a single synthetic demographic record.
        """
        col_names = self.get_column_names()
        rows = self._pick_demographic_rows(samples)

        demographics = []
        for i in range(samples):
            row_dict = dict(zip(col_names, rows[i]))
            
            age = row_dict["age"]
            sex = row_dict["sex"]  # 1=M, 2=F
            race = row_dict["race"]
            origin = row_dict["origin"]
            state_name = row_dict["name"]

            # Estimate birth year (relative to 2023)
            birth_year = 2023 - age
            birth_month, birth_day = self.get_birth_date()
            birthdate_str = f"{birth_year:04d}-{birth_month:02d}-{birth_day:02d}"

            # State abbreviation/code
            state_abbr = self._get_state_abbr(state_name)
            state_code = self._get_state_code(state_name)

            # Pick names
            first_name = self._pick_first_name(sex, birth_year, race, origin, state_abbr)
            last_name = self._pick_last_name(sex, birth_year, race, origin, state_abbr)

            # Education & occupation
            edu_occ = self.generate_education_and_occupation(age, state_code)
            is_student = edu_occ["is_student"]
            education_level = edu_occ["education_level"]
            occupation_title = edu_occ["occupation_title"]

            # Labor force / employment
            is_in_labor_force, is_employed = self.generate_employment_status(age, state_code, education_level)
            if not is_employed and is_in_labor_force:
                occupation_title = "Unemployed"
            elif not is_in_labor_force:
                occupation_title = None

            hobbies = self.get_hobbies(age, sex)
            aspirations = self.get_aspirations(age, sex)
            values = self.get_values(age, sex)

            sex_string = "male"
            race_string = "white"

            if sex == 2:
                sex_string = "female"

            if origin == 2:
                race_string = "hispanic"
            elif race == 2:
                race_string = "black"
            elif race == 4:
                race_string = "asian"
            

            demographic_record = DemographicRecord(
                first_name=first_name,
                last_name= last_name,
                state= state_name,
                age= age,
                birthdate= birthdate_str,
                sex= sex_string,
                race= race_string,
                is_student= is_student,
                education= education_level,
                is_in_labor_force= is_in_labor_force,
                is_employed= is_employed,
                occupation_category= occupation_title,
                hobbies= hobbies,
                aspirations= aspirations,
                values= values
            )
            
            demographics.append(demographic_record)

        return demographics


if __name__ == "__main__":
    demographic_generator = DemographicGenerator(db_path=database_name)

    output_file = "synthetic_demographics/synthetic_demographics.jsonl"
    batch_size = 10
    batch_count = 10

    for batch in range(batch_count):
        print(f"Batch {batch + 1}")
        # Example: Generate 10k demographic records

        start_time = time.time()
        results = demographic_generator.generate_demographic_batch(samples=batch_size)
        end_time = time.time()

        print("~" * 50)
        print(f"Generated {batch_size} demographics in {end_time - start_time:.4f} seconds.")
        print(f"Average time per record: {(end_time - start_time) / batch_size:.6f} seconds.")
        print("~" * 50)

        with open(output_file, "a") as f:
            for record in results:
                f.write(record.model_dump_json())
                f.write("\n")
    demographic_generator.db.close()