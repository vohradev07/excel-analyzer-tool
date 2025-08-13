import streamlit as st
import pandas as pd
import os
from fuzzywuzzy import fuzz, process
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Excel/CSV Data Analyzer",
    page_icon="📊",
    layout="wide"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'grouped_data' not in st.session_state:
    st.session_state.grouped_data = None
if 'loaded_files' not in st.session_state:
    st.session_state.loaded_files = []


class StreamlitExcelAnalyzer:
    def __init__(self):
        self.abbreviations = {
            'st': 'street', 'str': 'street', 'street': 'st',
            'rd': 'road', 'road': 'rd',
            'ave': 'avenue', 'av': 'avenue', 'avenue': 'ave',
            'blvd': 'boulevard', 'boulevard': 'blvd',
            'ln': 'lane', 'lane': 'ln',
            'dr': 'drive', 'drive': 'dr',
            'ct': 'court', 'court': 'ct',
            'pl': 'place', 'place': 'pl',
            'sq': 'square', 'square': 'sq',
            'cir': 'circle', 'circle': 'cir',
            'pkwy': 'parkway', 'parkway': 'pkwy',
            'apt': 'apartment', 'apartment': 'apt',
            'bldg': 'building', 'building': 'bldg',
            'chs': 'cooperative housing society', 'cooperative housing society': 'chs',
            'soc': 'society', 'society': 'soc',
            'twp': 'township', 'township': 'twp',
            'res': 'residence', 'residence': 'res',
            'cmplx': 'complex', 'complex': 'cmplx',
            'n': 'north', 'north': 'n',
            's': 'south', 'south': 's',
            'e': 'east', 'east': 'e',
            'w': 'west', 'west': 'w',
            'ne': 'northeast', 'northeast': 'ne',
            'nw': 'northwest', 'northwest': 'nw',
            'se': 'southeast', 'southeast': 'se',
            'sw': 'southwest', 'southwest': 'sw',
            'nr': 'near', 'near': 'nr',
            'opp': 'opposite', 'opposite': 'opp',
            'jn': 'junction', 'junction': 'jn',
            'xing': 'crossing', 'crossing': 'xing',
            'mall': 'shopping center', 'shopping center': 'mall',
            'ctr': 'center', 'center': 'ctr', 'centre': 'center',
            'mkt': 'market', 'market': 'mkt',
            'nagar': 'city', 'city': 'nagar',
            'gdn': 'garden', 'garden': 'gdn', 'gardens': 'gdn',
            'pk': 'park', 'park': 'pk',
            'hts': 'heights', 'heights': 'hts',
            'vly': 'valley', 'valley': 'vly',
            'hbr': 'harbor', 'harbor': 'hbr', 'harbour': 'hbr',
            'mdws': 'meadows', 'meadows': 'mdws',
            'plz': 'plaza', 'plaza': 'plz'
        }

    def normalize_text_for_matching(self, text):
        if not text or pd.isna(text):
            return ""
        words = str(text).lower().split()
        normalized_words = []
        for word in words:
            clean_word = word.strip('.,()[]{}:;-_')
            if clean_word in self.abbreviations:
                normalized_words.append(clean_word)
                normalized_words.append(self.abbreviations[clean_word])
            else:
                normalized_words.append(clean_word)
        return ' '.join(normalized_words)

    def create_searchable_variants(self, name):
        variants = set()
        variants.add(name.lower().strip())
        normalized = self.normalize_text_for_matching(name)
        variants.add(normalized)
        words = name.lower().split()
        for i, word in enumerate(words):
            clean_word = word.strip('.,()[]{}:;-_')
            if clean_word in self.abbreviations:
                variant_words = words.copy()
                variant_words[i] = self.abbreviations[clean_word]
                variants.add(' '.join(variant_words))
        return list(variants)

    def enhanced_fuzzy_match(self, search_term, name, threshold=70):
        search_normalized = self.normalize_text_for_matching(search_term)
        search_variants = [search_term.lower().strip(), search_normalized]
        name_variants = self.create_searchable_variants(name)
        best_score = 0
        best_match_info = None
        for search_var in search_variants:
            for name_var in name_variants:
                scores = [
                    fuzz.ratio(search_var, name_var),
                    fuzz.partial_ratio(search_var, name_var),
                    fuzz.token_sort_ratio(search_var, name_var),
                    fuzz.token_set_ratio(search_var, name_var)
                ]
                max_score = max(scores)
                if max_score > best_score:
                    best_score = max_score
                    best_match_info = {
                        'search_variant': search_var,
                        'name_variant': name_var,
                        'score_type': ['ratio', 'partial_ratio', 'token_sort_ratio', 'token_set_ratio'][scores.index(max_score)]
                    }
        return best_score, best_match_info

    def get_hood_name_for_society(self, society_name, df):
        if df is None or 'Hood Name' not in df.columns:
            return None
        rows = df[df['Society Name'] == society_name]
        if not rows.empty:
            return rows['Hood Name'].iloc[0]
        return None

    def fuzzy_match_societies(self, search_term, df, grouped_data, threshold=70):
        matching_societies = []
        for society in grouped_data.groups.keys():
            score, match_info = self.enhanced_fuzzy_match(search_term, society, threshold)
            if score >= threshold:
                hood_name = self.get_hood_name_for_society(society, df)
                matching_societies.append((society, score, hood_name, match_info))
        hood_groups = {}
        for society, score, hood_name, match_info in matching_societies:
            if hood_name:
                hood_groups.setdefault(hood_name, []).append((society, score, match_info))
        verified_matches = []
        for hood_name, matches in hood_groups.items():
            matches.sort(key=lambda x: x[1], reverse=True)
            for society, score, match_info in matches:
                verified_matches.append((society, score, hood_name, match_info))
        verified_matches.sort(key=lambda x: x[1], reverse=True)
        return verified_matches


def load_files(uploaded_files):
    if not uploaded_files:
        return None, None
    all_dfs = []
    for uploaded_file in uploaded_files:
        try:
            if uploaded_file.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            df.columns = df.columns.str.strip()
            if 'Society Name' not in df.columns:
                st.warning(f"Column 'Society Name' not found in {uploaded_file.name}. Skipping.")
                continue
            df['Society Name'] = df['Society Name'].astype(str).str.strip()
            df = df[df['Society Name'] != 'nan']
            df['Source_File'] = uploaded_file.name
            all_dfs.append(df)
        except Exception as e:
            st.warning(f"Failed to load {uploaded_file.name}: {str(e)}")
    if not all_dfs:
        return None, None
    combined_df = pd.concat(all_dfs, ignore_index=True)
    grouped_data = combined_df.groupby('Society Name')
    return combined_df, grouped_data


def main():
    st.title("📊 Excel/CSV Data Analyzer")
    analyzer = StreamlitExcelAnalyzer()

    st.sidebar.header("File Upload")
    uploaded_files = st.sidebar.file_uploader("Choose Excel or CSV files", type=['xlsx', 'xls', 'csv'], accept_multiple_files=True)
    if uploaded_files and st.sidebar.button("Load Data"):
        df, grouped_data = load_files(uploaded_files)
        if df is not None and grouped_data is not None:
            st.session_state.df = df
            st.session_state.grouped_data = grouped_data
            st.session_state.loaded_files = [f.name for f in uploaded_files]
            st.success(f"✅ Loaded {len(uploaded_files)} files, {len(df)} rows, {len(grouped_data.groups)} unique societies")
        else:
            st.error("❌ No files could be loaded")

    if st.sidebar.button("Clear All Data"):
        st.session_state.df = None
        st.session_state.grouped_data = None
        st.session_state.loaded_files = []
        st.success("Data cleared")

    if st.session_state.df is not None:
        st.header("🔍 Search")
        search_mode = st.radio("Search by:", ["Society Name", "Hood Name"], horizontal=True)
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            search_term = st.text_input(f"Search for {search_mode.lower()}:", placeholder=f"Enter {search_mode.lower()}...")
        with col2:
            fuzzy_enabled = st.checkbox("Enable Fuzzy Matching", value=True)
        with col3:
            similarity_threshold = st.slider("Similarity %", 50, 95, 70) if fuzzy_enabled else 70

        if search_term:
            df = st.session_state.df
            grouped_data = st.session_state.grouped_data
            if fuzzy_enabled:
                if search_mode == "Society Name":
                    matches = analyzer.fuzzy_match_societies(search_term, df, grouped_data, similarity_threshold)
                else:
                    matches = []
                    if "Hood Name" in df.columns:
                        for hood in df["Hood Name"].dropna().unique():
                            score, match_info = analyzer.enhanced_fuzzy_match(search_term, hood, similarity_threshold)
                            if score >= similarity_threshold:
                                for society in df[df["Hood Name"] == hood]["Society Name"].unique():
                                    matches.append((society, score, hood, match_info))
                        matches.sort(key=lambda x: x[1], reverse=True)
                    else:
                        st.warning("No 'Hood Name' column found")
                verified_matches = matches
                if verified_matches:
                    st.subheader(f"Found {len(verified_matches)} matches")
                    match_options = [(f"{soc} ({len(grouped_data.get_group(soc))} activities) - {score}% in {hood}", soc)
                                     for soc, score, hood, _ in verified_matches]
                    selected_match = st.selectbox("Select a society to view details:", [m[0] for m in match_options])
                    selected_society = next(s for t, s in match_options if t == selected_match)
                    society_data = grouped_data.get_group(selected_society)
                    st.dataframe(society_data, use_container_width=True)
                else:
                    st.warning("No matches found")
            else:
                if search_mode == "Society Name":
                    matching_societies = [soc for soc in grouped_data.groups.keys() if search_term.lower() in soc.lower()]
                else:
                    if "Hood Name" in df.columns:
                        matching_hoods = [hood for hood in df["Hood Name"].dropna().unique() if search_term.lower() in hood.lower()]
                        matching_societies = df[df["Hood Name"].isin(matching_hoods)]["Society Name"].unique().tolist()
                    else:
                        st.warning("No 'Hood Name' column found")
                        matching_societies = []
                if matching_societies:
                    selected_society = st.selectbox("Select a society to view details:", matching_societies)
                    society_data = grouped_data.get_group(selected_society)
                    st.dataframe(society_data, use_container_width=True)
                else:
                    st.warning("No matches found")


if __name__ == "__main__":
    main()
