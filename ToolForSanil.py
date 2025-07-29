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
        # Abbreviation mappings for fuzzy matching
        self.abbreviations = {
            # Common street/location abbreviations
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

            # Building/housing abbreviations
            'apt': 'apartment', 'apartment': 'apt',
            'bldg': 'building', 'building': 'bldg',
            'chs': 'cooperative housing society', 'cooperative housing society': 'chs',
            'soc': 'society', 'society': 'soc',
            'twp': 'township', 'township': 'twp',
            'res': 'residence', 'residence': 'res',
            'cmplx': 'complex', 'complex': 'cmplx',

            # Direction abbreviations
            'n': 'north', 'north': 'n',
            's': 'south', 'south': 's',
            'e': 'east', 'east': 'e',
            'w': 'west', 'west': 'w',
            'ne': 'northeast', 'northeast': 'ne',
            'nw': 'northwest', 'northwest': 'nw',
            'se': 'southeast', 'southeast': 'se',
            'sw': 'southwest', 'southwest': 'sw',

            # Other common abbreviations
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
        """Normalize text by expanding abbreviations for better matching"""
        if not text or pd.isna(text):
            return ""

        # Convert to lowercase and split into words
        words = str(text).lower().split()
        normalized_words = []

        for word in words:
            # Remove common punctuation
            clean_word = word.strip('.,()[]{}:;-_')

            # Check if word is an abbreviation and expand it
            if clean_word in self.abbreviations:
                # Add both the abbreviation and its expansion for better matching
                normalized_words.append(clean_word)
                normalized_words.append(self.abbreviations[clean_word])
            else:
                normalized_words.append(clean_word)

        return ' '.join(normalized_words)

    def create_searchable_variants(self, society_name):
        """Create multiple searchable variants of a society name"""
        variants = set()

        # Original name
        variants.add(society_name.lower().strip())

        # Normalized version with abbreviations expanded
        normalized = self.normalize_text_for_matching(society_name)
        variants.add(normalized)

        # Create variants by replacing abbreviations
        words = society_name.lower().split()
        for i, word in enumerate(words):
            clean_word = word.strip('.,()[]{}:;-_')
            if clean_word in self.abbreviations:
                # Create variant with abbreviation replaced
                variant_words = words.copy()
                variant_words[i] = self.abbreviations[clean_word]
                variants.add(' '.join(variant_words))

        return list(variants)

    def enhanced_fuzzy_match(self, search_term, society_name, threshold=70):
        """Enhanced fuzzy matching that considers abbreviations"""
        # Normalize search term
        search_normalized = self.normalize_text_for_matching(search_term)
        search_variants = [search_term.lower().strip(), search_normalized]

        # Get variants of society name
        society_variants = self.create_searchable_variants(society_name)

        best_score = 0
        best_match_info = None

        # Test all combinations of search variants against society variants
        for search_var in search_variants:
            for society_var in society_variants:
                # Try different fuzzy matching algorithms
                scores = [
                    fuzz.ratio(search_var, society_var),
                    fuzz.partial_ratio(search_var, society_var),
                    fuzz.token_sort_ratio(search_var, society_var),
                    fuzz.token_set_ratio(search_var, society_var)
                ]

                max_score = max(scores)
                if max_score > best_score:
                    best_score = max_score
                    best_match_info = {
                        'search_variant': search_var,
                        'society_variant': society_var,
                        'score_type': ['ratio', 'partial_ratio', 'token_sort_ratio', 'token_set_ratio'][
                            scores.index(max_score)]
                    }

        return best_score, best_match_info

    def get_hood_name_for_society(self, society_name, df):
        """Get the Hood Name for a given society"""
        if df is None:
            return None

        # Check if Hood Name column exists
        if 'Hood Name' not in df.columns:
            return None

        # Find the hood name for this society
        society_rows = df[df['Society Name'] == society_name]
        if not society_rows.empty:
            return society_rows['Hood Name'].iloc[0]
        return None

    def fuzzy_match_societies(self, search_term, df, grouped_data, threshold=70):
        """Find societies using enhanced fuzzy matching with abbreviation support and Hood Name verification"""
        if grouped_data is None:
            return []

        matching_societies = []
        society_names = list(grouped_data.groups.keys())

        # Use enhanced fuzzy matching for each society
        for society in society_names:
            score, match_info = self.enhanced_fuzzy_match(search_term, society, threshold)
            if score >= threshold:
                hood_name = self.get_hood_name_for_society(society, df)
                matching_societies.append((society, score, hood_name, match_info))

        # Group by Hood Name for verification
        hood_groups = {}
        for society, score, hood_name, match_info in matching_societies:
            if hood_name:
                if hood_name not in hood_groups:
                    hood_groups[hood_name] = []
                hood_groups[hood_name].append((society, score, match_info))

        # Verify matches within the same Hood Name
        verified_matches = []
        for hood_name, matches in hood_groups.items():
            if len(matches) >= 1:  # At least one match in this hood
                # Sort by score (highest first)
                matches.sort(key=lambda x: x[1], reverse=True)

                # Add all matches from this hood (they're all verified by being in same hood)
                for society, score, match_info in matches:
                    verified_matches.append((society, score, hood_name, match_info))

        # Sort all verified matches by score
        verified_matches.sort(key=lambda x: x[1], reverse=True)

        return verified_matches


def load_files(uploaded_files):
    """Load and aggregate data from uploaded files"""
    if not uploaded_files:
        return None, None

    all_dataframes = []
    loaded_file_info = []

    for uploaded_file in uploaded_files:
        try:
            # Load file based on extension
            if uploaded_file.name.lower().endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            # Clean column names (remove extra spaces)
            df.columns = df.columns.str.strip()

            # Check if Society Name column exists
            if 'Society Name' not in df.columns:
                st.warning(f"Column 'Society Name' not found in {uploaded_file.name}. Skipping this file.")
                continue

            # Clean Society Name data
            df['Society Name'] = df['Society Name'].astype(str).str.strip()
            df = df[df['Society Name'] != 'nan']  # Remove rows where Society Name is NaN

            # Add source file column for tracking
            df['Source_File'] = uploaded_file.name

            all_dataframes.append(df)
            loaded_file_info.append((uploaded_file.name, len(df)))

        except Exception as e:
            st.warning(f"Failed to load {uploaded_file.name}: {str(e)}")
            continue

    if not all_dataframes:
        return None, None

    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    # Group data by Society Name
    grouped_data = combined_df.groupby('Society Name')

    return combined_df, grouped_data


def main():
    st.title("📊 Excel/CSV Data Analyzer")
    st.markdown("Upload Excel or CSV files to analyze society data with fuzzy search capabilities")

    analyzer = StreamlitExcelAnalyzer()

    # Sidebar for file upload
    st.sidebar.header("File Upload")
    uploaded_files = st.sidebar.file_uploader(
        "Choose Excel or CSV files",
        type=['xlsx', 'xls', 'csv'],
        accept_multiple_files=True,
        help="Upload one or more Excel or CSV files containing society data"
    )

    if uploaded_files:
        if st.sidebar.button("Load Data"):
            with st.spinner("Loading files..."):
                df, grouped_data = load_files(uploaded_files)

                if df is not None and grouped_data is not None:
                    st.session_state.df = df
                    st.session_state.grouped_data = grouped_data
                    st.session_state.loaded_files = [f.name for f in uploaded_files]

                    total_rows = len(df)
                    unique_societies = len(grouped_data.groups)
                    files_count = len(uploaded_files)

                    st.success(
                        f"✅ Loaded {files_count} files: {total_rows} total rows with {unique_societies} unique societies")
                else:
                    st.error("❌ No files could be loaded successfully")

    # Clear data button
    if st.sidebar.button("Clear All Data"):
        st.session_state.df = None
        st.session_state.grouped_data = None
        st.session_state.loaded_files = []
        st.success("All data cleared")

    # Display loaded files info
    if st.session_state.loaded_files:
        st.sidebar.subheader("Loaded Files:")
        for file_name in st.session_state.loaded_files:
            st.sidebar.text(f"📄 {file_name}")

    # Main content area
    if st.session_state.df is not None and st.session_state.grouped_data is not None:
        # Search section
        st.header("🔍 Search Societies")

        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            search_term = st.text_input("Search for society name:", placeholder="Enter society name...")

        with col2:
            fuzzy_enabled = st.checkbox("Enable Fuzzy Matching", value=True)

        with col3:
            if fuzzy_enabled:
                similarity_threshold = st.slider("Similarity %", 50, 95, 70)
            else:
                similarity_threshold = 70

        # Search results
        if search_term:
            with st.spinner("Searching..."):
                if fuzzy_enabled:
                    # Use fuzzy matching
                    verified_matches = analyzer.fuzzy_match_societies(
                        search_term, st.session_state.df, st.session_state.grouped_data, similarity_threshold
                    )

                    if verified_matches:
                        st.subheader(
                            f"Found {len(verified_matches)} fuzzy matches (threshold: {similarity_threshold}%)")

                        # Create selection options
                        match_options = []
                        for society, score, hood_name, match_info in verified_matches:
                            count = len(st.session_state.grouped_data.get_group(society))
                            match_type = ""
                            if match_info and 'score_type' in match_info:
                                if 'token' in match_info['score_type']:
                                    match_type = " (abbrev.)"

                            display_text = f"{society} ({count} activities) - {score}% match{match_type} in {hood_name}"
                            match_options.append((display_text, society))

                        selected_match = st.selectbox(
                            "Select a society to view details:",
                            options=[option[0] for option in match_options],
                            index=0
                        )

                        # Find the selected society name
                        selected_society = None
                        for display_text, society_name in match_options:
                            if display_text == selected_match:
                                selected_society = society_name
                                break

                        if selected_society:
                            # Display society data
                            st.subheader(f"Data for: {selected_society}")
                            society_data = st.session_state.grouped_data.get_group(selected_society)

                            # Show file breakdown
                            if 'Source_File' in society_data.columns:
                                file_counts = society_data['Source_File'].value_counts()
                                st.info(f"📊 {len(society_data)} activities from files: " +
                                        ", ".join([f"{file}: {count}" for file, count in file_counts.items()]))

                            # Display the data table
                            st.dataframe(society_data, use_container_width=True)

                            # Download button for filtered data
                            csv = society_data.to_csv(index=False)
                            st.download_button(
                                label="📥 Download filtered data as CSV",
                                data=csv,
                                file_name=f"{selected_society}_data.csv",
                                mime="text/csv"
                            )
                    else:
                        st.warning(f"No fuzzy matches found (threshold: {similarity_threshold}%)")
                else:
                    # Use exact matching
                    search_term_lower = search_term.lower()
                    matching_societies = []
                    for society_name in st.session_state.grouped_data.groups.keys():
                        if search_term_lower in society_name.lower():
                            matching_societies.append(society_name)

                    if matching_societies:
                        st.subheader(f"Found {len(matching_societies)} exact matches")

                        selected_society = st.selectbox(
                            "Select a society to view details:",
                            options=matching_societies
                        )

                        if selected_society:
                            # Display society data
                            st.subheader(f"Data for: {selected_society}")
                            society_data = st.session_state.grouped_data.get_group(selected_society)

                            # Show file breakdown
                            if 'Source_File' in society_data.columns:
                                file_counts = society_data['Source_File'].value_counts()
                                st.info(f"📊 {len(society_data)} activities from files: " +
                                        ", ".join([f"{file}: {count}" for file, count in file_counts.items()]))

                            # Display the data table
                            st.dataframe(society_data, use_container_width=True)

                            # Download button for filtered data
                            csv = society_data.to_csv(index=False)
                            st.download_button(
                                label="📥 Download filtered data as CSV",
                                data=csv,
                                file_name=f"{selected_society}_data.csv",
                                mime="text/csv"
                            )
                    else:
                        st.warning("No matching societies found")

        # Show all societies option
        if st.button("Show All Societies"):
            st.subheader(f"All Societies ({len(st.session_state.grouped_data.groups)})")

            # Create a summary dataframe
            society_summary = []
            for society_name, group in st.session_state.grouped_data:
                activity_count = len(group)
                files = group['Source_File'].unique() if 'Source_File' in group.columns else ['Unknown']
                society_summary.append({
                    'Society Name': society_name,
                    'Activity Count': activity_count,
                    'Source Files': ', '.join(files)
                })

            summary_df = pd.DataFrame(society_summary)
            st.dataframe(summary_df, use_container_width=True)

            # Download button for summary
            csv = summary_df.to_csv(index=False)
            st.download_button(
                label="📥 Download society summary as CSV",
                data=csv,
                file_name="society_summary.csv",
                mime="text/csv"
            )

    else:
        st.info("👆 Please upload Excel or CSV files using the sidebar to get started")

        # Show example of expected data format
        st.subheader("Expected Data Format")
        st.markdown("""
        Your Excel/CSV files should contain at least these columns:
        - **Society Name**: The name of the society/organization
        - **Hood Name** (optional): For better fuzzy matching verification
        - Other columns with relevant data

        The tool will automatically:
        - Combine data from multiple files
        - Handle abbreviations in society names
        - Provide fuzzy search with customizable similarity thresholds
        - Track which file each record came from
        """)


if __name__ == "__main__":
    main()
