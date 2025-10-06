import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.power import TTestIndPower, NormalIndPower
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.proportion as prop
import ast

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

COLOR_PALETTE = {
    'control': '#7FB3D5',  # Soft blue
    'variant_b': '#F8C471',  # Soft orange
    'variant_c': '#82E0AA',  # Soft green
    'desktop': '#85C1E9',   # Light blue
    'mobile': '#F9E79F',    # Light yellow
    'plan_1_month': '#AED6F1',  # Very light blue
    'plan_12_month': '#FAD7A0', # Very light orange
    'plan_24_month': '#A9DFBF', # Very light green
    'normal_sessions': '#D6EAF8',  # Pale blue
    'outliers': '#F5B7B1'   # Soft red
}

# Variant colors for bar charts
VARIANT_COLORS = [COLOR_PALETTE['control'], COLOR_PALETTE['variant_b'], COLOR_PALETTE['variant_c']]
PLAN_COLORS = [COLOR_PALETTE['plan_1_month'], COLOR_PALETTE['plan_12_month'], COLOR_PALETTE['plan_24_month']]


def parse_and_filter_subscriptions(sub_data):
    """
    Parse subscription data and filter for 1, 12, or 24-month plans only.
    Handles both dictionary format and string representations of dictionaries.
    Returns filtered subscription data and total count of relevant subscriptions.
    """
    if pd.isna(sub_data):
        return {}, 0

    sub_dict = {}
    if isinstance(sub_data, str) and sub_data.startswith('{'):
        try:
            sub_dict = ast.literal_eval(sub_data)
        except:
            return {}, 0 # Return empty if string is not a valid dict
    elif isinstance(sub_data, dict):
        sub_dict = sub_data

    new_sub_data = {}
    relevant_count = 0
    for plan_key, count in sub_dict.items():
        plan_str = str(plan_key).lower().strip()
        if '12' in plan_str or 'annual' in plan_str:
            new_sub_data[plan_key] = count
            relevant_count += count
        elif '24' in plan_str or '2year' in plan_str:
            new_sub_data[plan_key] = count
            relevant_count += count
        elif '1' in plan_str or 'monthly' in plan_str:
            new_sub_data[plan_key] = count
            relevant_count += count

    return new_sub_data, relevant_count

def calculate_confidence_intervals(conversions, sample_size):
    """Calculate Wilson confidence intervals for binomial proportions."""
    lower, upper = proportion_confint(conversions, sample_size, alpha=0.05, method='wilson')
    return lower, upper

def calculate_effect_size(control_success, control_total, treatment_success, treatment_total):
    """Calculate Cohen's h effect size for comparing two proportions."""
    p_control = control_success / control_total
    p_treatment = treatment_success / treatment_total
    effect_size = 2 * (np.arcsin(np.sqrt(p_treatment)) - np.arcsin(np.sqrt(p_control)))
    return effect_size

def run_power_analysis(control_size, treatment_size, baseline_rate=0.015, mde=0.002, alpha=0.05):
    """
    Perform power analysis to check if sample sizes are sufficient.
    Calculates required sample size and achieved power for detecting the specified effect.
    """
    effect_size = prop.proportion_effectsize(baseline_rate, baseline_rate + mde)
    power_analysis = NormalIndPower()
    
    required_sample = power_analysis.solve_power(
        effect_size=effect_size,
        power=0.8,
        alpha=alpha,
        ratio=1 # Assuming equal size groups
    )
    
    actual_power = power_analysis.solve_power(
        effect_size=effect_size,
        nobs1=min(control_size, treatment_size),
        alpha=alpha,
        ratio=1
    )
    
    print(f"Required sample per group: {required_sample:.0f}")
    print(f"Actual samples: Control={control_size}, Treatment={treatment_size}")
    print(f"Achieved power: {actual_power:.1%}")
    
    return required_sample, min(control_size, treatment_size) >= required_sample

def analyze_plan_mix_exclusive(df_group, group_name):
    """
    Analyze distribution of 1, 12, and 24-month subscription plans for a specific group.
    Prints the plan mix percentages and returns counts and total subscriptions.
    """
    plan_counts = {'1_month': 0, '12_month': 0, '24_month': 0}

    for sub_dict in df_group['subscriptionsTotalByCycle'].dropna():
        for plan_key, count in sub_dict.items():
            plan_str = str(plan_key).lower().strip()
            if '12' in plan_str or 'annual' in plan_str:
                plan_counts['12_month'] += count
            elif '24' in plan_str or '2year' in plan_str:
                plan_counts['24_month'] += count
            elif '1' in plan_str or 'monthly' in plan_str:
                plan_counts['1_month'] += count

    total_subscriptions = sum(plan_counts.values())

    print(f"\n{group_name} Subscription Plan Mix (1/12/24 months ONLY):")
    print(f"  Total Relevant Subscriptions: {total_subscriptions}")
    
    if total_subscriptions > 0:
        for plan, count in plan_counts.items():
            percentage = (count / total_subscriptions) * 100
            print(f"  {plan}: {count} ({percentage:.1f}%)")
    else:
        print(f"  No relevant 1/12/24 month subscriptions found")

    return plan_counts, total_subscriptions

def filter_dataset_to_1_12_24_only(df_clean):
    """
    Filter the dataset to include only 1, 12, and 24-month subscription plans.
    Updates subscription counts and removes non-relevant plan data.
    """
    print("\nFiltering dataset to only include 1, 12, and 24-month plans...")
    df_filtered = df_clean.copy()

    results = df_filtered['subscriptionsTotalByCycle'].apply(parse_and_filter_subscriptions)

    df_filtered['subscriptionsTotalByCycle'] = [res[0] for res in results]
    df_filtered['subscriptionsTotal'] = [res[1] for res in results]
    
    df_filtered['subscriptionsTotalByCycle'] = df_filtered['subscriptionsTotalByCycle'].apply(lambda d: d if d else np.nan)

    print(f"Filtered total subscriptions (1/12/24 only): {df_filtered['subscriptionsTotal'].sum()}")

    return df_filtered

def load_and_validate_data():
    """
    Load and validate the experiment data.
    Handles duplicate sessions and ensures data integrity across key dimensions.
    """
    print("1. LOADING AND VALIDATING DATA")
    print("-" * 40)
    
    df = pd.read_csv('SecureNetMarketingCaseStudy.csv')
    print(f"Original dataset shape: {df.shape}")
    
    # Data integrity validation
    duplicate_session_ids = df[df.duplicated(subset=['sessionId'], keep=False)]
    num_split_sessions = duplicate_session_ids['sessionId'].nunique()
    print(f"Found {num_split_sessions} sessionIDs that appear on multiple days.")

    if num_split_sessions > 0:
        consistency_check = duplicate_session_ids.groupby('sessionId').agg({
            'deviceType': 'nunique',
            'flag': 'nunique',
            'country': 'nunique'
        }).rename(columns={
            'deviceType': 'unique_devices',
            'flag': 'unique_flags',
            'country': 'unique_countries' 
        })
        
        inconsistent_devices = consistency_check[consistency_check['unique_devices'] > 1]
        inconsistent_flags = consistency_check[consistency_check['unique_flags'] > 1]
        inconsistent_countries = consistency_check[consistency_check['unique_countries'] > 1]
        
        problematic_sessions = set(inconsistent_devices.index) | set(inconsistent_flags.index) | set(inconsistent_countries.index)
        
        if len(problematic_sessions) > 0:
            print("Removing problematic sessions with data integrity issues.")
            df = df[~df['sessionId'].isin(problematic_sessions)]
            duplicate_session_ids = df[df.duplicated(subset=['sessionId'], keep=False)]
            num_split_sessions = duplicate_session_ids['sessionId'].nunique()

        if num_split_sessions > 0:
            consolidation_dict = {
                'flag': 'first',
                'Date': 'min',
                'country': 'first',
                'deviceType': 'first',
                'signups': 'sum',
                'sessionDurationInSeconds': 'sum',
                'PageViews': 'sum',
                'subscriptionsTotal': 'sum',
                'subscriptionBillingsConvertedInCHFTotal': 'sum'
            }
            
            consolidated_duplicates = duplicate_session_ids.groupby('sessionId').agg(consolidation_dict).reset_index()
            
            def merge_sub_dicts(series):
                dicts = [d for d in series if isinstance(d, dict) and d != {}]
                return dicts[0] if dicts else {}

            consolidated_subs = duplicate_session_ids.groupby('sessionId')['subscriptionsTotalByCycle'].apply(merge_sub_dicts).reset_index()
            consolidated_duplicates = consolidated_duplicates.merge(consolidated_subs, on='sessionId', how='left')
            
            single_sessions_df = df[~df.duplicated(subset=['sessionId'], keep=False)]
            df = pd.concat([single_sessions_df, consolidated_duplicates], ignore_index=True)
            print(f"Consolidated {num_split_sessions} split sessions. New dataframe shape: {df.shape}")
    
    return df

def detect_and_remove_outliers(df):
    """
    Detect and remove outlier sessions using a CONSERVATIVE approach.
    This revised function only targets obvious bot-like activity to avoid removing
    legitimate, high-engagement user sessions.
    """
    print("\n2. OUTLIER DETECTION AND REMOVAL")
    print("-" * 40)
    
    numeric_columns = ['signups', 'sessionDurationInSeconds', 'PageViews', 
                       'subscriptionsTotal', 'subscriptionBillingsConvertedInCHFTotal']
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['country'] = df['country'].fillna('Unknown')
    df['pages_per_second'] = df['PageViews'] / df['sessionDurationInSeconds'].replace(0, np.nan)
    
    # --- REVISED CONSERVATIVE OUTLIER LOGIC ---
    # This new logic is designed to be highly specific and avoid removing valid users.
    # It targets only clear, non-human patterns.

    # Rule 1: Impossible page view speed, indicating a script or bot.
    # A human cannot reasonably view more than 5 pages per second.
    bot_speed_threshold = 5 
    speed_outliers = df['pages_per_second'] > bot_speed_threshold
    
    # Rule 2: Zero-duration sessions with multiple page views, another clear bot signal.
    zero_duration_outliers = (df['sessionDurationInSeconds'] == 0) & (df['PageViews'] > 1)
    
    # Combine only the conservative outlier rules.
    outlier_mask = speed_outliers | zero_duration_outliers
    
    print(f"Applying conservative filter to remove obvious bot activity.")
    print(f"Removing {outlier_mask.sum()} outlier sessions ({outlier_mask.sum()/len(df)*100:.2f}% of data)")
    df_clean = df[~outlier_mask].copy()
    
    # Create conversion columns
    df_clean['has_subscription'] = (df_clean['subscriptionsTotal'] > 0).astype(int)
    df_clean['has_signup'] = (df_clean['signups'] > 0).astype(int)
    df_clean['is_engaged'] = (df_clean['PageViews'] >= 3).astype(int)
    
    print(f"Clean dataset: {len(df_clean):,} sessions")
    print(f"Sessions by plan order: {df_clean['flag'].value_counts().to_dict()}")
    
    return df_clean

def run_statistical_tests(df_filtered):
    """
    Perform statistical comparison between control and treatment variants.
    Uses proportion Z-tests with Bonferroni correction for multiple comparisons.
    Calculates confidence intervals and effect sizes for comprehensive analysis.
    """
    print("\n4. STATISTICAL SIGNIFICANCE TESTING")
    print("-" * 40)
    
    # --- CORRECTED: Define Control and Treatments explicitly ---
    control_flag = 'A'  # True Control Group: Plan order 1/24/12
    treatment_flags = ['B', 'C'] # Treatment Groups
    
    results = []
    alpha = 0.05
    bonferroni_alpha = alpha / len(treatment_flags) # For two comparisons
    
    control_data = df_filtered[df_filtered['flag'] == control_flag]
    control_size = len(control_data)
    control_conversion = control_data['has_subscription'].mean()

    for treatment in treatment_flags:
        treatment_data = df_filtered[df_filtered['flag'] == treatment]
        treatment_size = len(treatment_data)
        
        print(f"\nANALYSIS: {control_flag} (Control: 1/24/12) vs {treatment} (Treatment)")
        print(f"Sample sizes: {control_flag}={control_size:,}, {treatment}={treatment_size:,}")

        # Conversion rates
        treatment_conversion = treatment_data['has_subscription'].mean()
        conversion_lift = (treatment_conversion - control_conversion) / control_conversion if control_conversion > 0 else 0
        
        # Statistical tests for conversion
        count = np.array([treatment_data['has_subscription'].sum(), control_data['has_subscription'].sum()])
        nobs = np.array([treatment_size, control_size])
        # Note: Order is treatment vs control for z-test
        z_stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
        
        # Confidence intervals
        control_ci = calculate_confidence_intervals(control_data['has_subscription'].sum(), control_size)
        treatment_ci = calculate_confidence_intervals(treatment_data['has_subscription'].sum(), treatment_size)
        
        # Effect size
        effect_size = calculate_effect_size(
            control_data['has_subscription'].sum(), control_size,
            treatment_data['has_subscription'].sum(), treatment_size
        )
        
        results.append({
            'comparison': f'{control_flag} vs {treatment}',
            'control_size': control_size,
            'treatment_size': treatment_size,
            'conversion_control': control_conversion,
            'conversion_treatment': treatment_conversion,
            'conversion_lift': conversion_lift,
            'conversion_p_value': p_value,
            'control_ci': control_ci,
            'treatment_ci': treatment_ci,
            'effect_size': effect_size,
            'significant': p_value < bonferroni_alpha
        })
        
        print(f"Conversion Rates: {control_flag}={control_conversion:.4f}, {treatment}={treatment_conversion:.4f}")
        print(f"Lift: {conversion_lift:+.2%}")
        print(f"Z-test p-value: {p_value:.6f} {'(SIGNIFICANT)' if p_value < bonferroni_alpha else '(NOT SIGNIFICANT)'}")
        print(f"Bonferroni threshold: {bonferroni_alpha:.4f}")
        print(f"Control 95% CI: [{control_ci[0]:.4f}, {control_ci[1]:.4f}]")
        print(f"Treatment 95% CI: [{treatment_ci[0]:.4f}, {treatment_ci[1]:.4f}]")
        print(f"Effect size: {effect_size:.3f}")

    return results

def analyze_segments(df_clean):
    """
    Analyze conversion performance across different user segments.
    Examines geographic and device-based variations in conversion rates.
    """
    print("\n5. SEGMENTATION ANALYSIS")
    print("-" * 40)
    
    insights = []
    
    # Geographic analysis
    if 'country' in df_clean.columns:
        top_countries = df_clean['country'].value_counts().head(5).index
        df_clean['country_group'] = df_clean['country'].apply(
            lambda x: x if x in top_countries else 'Other'
        )
        
        geo_performance = df_clean.groupby(['country_group', 'flag'])['has_subscription'].mean().unstack()
        print("Performance by Country:")
        print(geo_performance.round(4))
        
    # Device type analysis
    if 'deviceType' in df_clean.columns:
        device_performance = df_clean.groupby(['deviceType', 'flag'])['has_subscription'].mean().unstack()
        print("\nPerformance by Device Type:")
        print(device_performance.round(4))
    
    return insights

def calculate_business_impact(df_filtered, results):
    """
    Calculate the business impact of the best performing treatment variant.
    Compares against control to determine additional conversions achieved.
    """
    print("\n6. BUSINESS IMPACT ANALYSIS")
    print("-" * 40)

    # Calculate metrics for all groups
    group_metrics = {}
    for flag in ['A', 'B', 'C']:
        group_data = df_filtered[df_filtered['flag'] == flag]
        group_metrics[flag] = {
            'size': len(group_data),
            'conversions': group_data['has_subscription'].sum(),
            'conversion_rate': group_data['has_subscription'].mean(),
        }
    
    control_group = group_metrics['A']
    
    # Find the best performing group from the TREATMENT variants only
    treatment_groups = {k: v for k, v in group_metrics.items() if k != 'A'}
    best_treatment_item = max(treatment_groups.items(), key=lambda x: x[1]['conversion_rate'])
    
    best_variant_name = best_treatment_item[0]
    best_variant_metrics = best_treatment_item[1]
    
    print(f"Control Conversion Rate (A): {control_group['conversion_rate']:.4f}")
    print(f"Best Treatment Variant: {best_variant_name}")
    print(f"Best Treatment Conversion Rate: {best_variant_metrics['conversion_rate']:.4f}")
    
    # Calculate additional conversions of the best treatment vs. control
    additional_conversions = (best_variant_metrics['conversion_rate'] - control_group['conversion_rate']) * best_variant_metrics['size']
    
    print(f"Additional conversions from {best_variant_name} vs. Control: {additional_conversions:.1f}")
    
    return {
        'best_group': best_variant_name,
        'conversion_lift': (best_variant_metrics['conversion_rate'] - control_group['conversion_rate']),
    }

def make_recommendation(results, business_impact):
    """
    Make final implementation recommendation based on statistical results.
    Considers both statistical significance and practical business impact.
    """
    print("\n7. FINAL RECOMMENDATION")
    print("-" * 40)
    
    best_result = max(results, key=lambda x: x['conversion_lift'])

    if best_result['significant'] and best_result['conversion_lift'] > 0:
        print(f"Recommend implementing Plan Order {best_result['comparison'].split()[-1]}")
        print(f"Rationale: Statistically significant improvement (p={best_result['conversion_p_value']:.6f})")
        print(f"Conversion lift: {best_result['conversion_lift']:+.2%}")
    else:
        print("Recommend maintaining current Plan Order A")
        print("Rationale: No variant showed statistically significant improvement over control")
        
def create_conversion_rate_plot(results):
    """
    Create bar chart comparing conversion rates across variants with confidence intervals.
    Includes lift annotations for easy comparison between treatment and control groups.
    Uses consistent, light color palette for professional appearance.
    """
    print("\n8. VISUALIZING RESULTS: PRIMARY CONVERSION RATES")
    print("-" * 40)
    
    control_res = results[0] 
    labels = [
        f"A (Control)\n{control_res['conversion_control']:.2%}", 
        f"Variant B\n{results[0]['conversion_treatment']:.2%}", 
        f"Variant C\n{results[1]['conversion_treatment']:.2%}"
    ]
    conv_rates = [
        control_res['conversion_control'], 
        results[0]['conversion_treatment'], 
        results[1]['conversion_treatment']
    ]
    
    ci_low = [
        control_res['control_ci'][0], 
        results[0]['treatment_ci'][0], 
        results[1]['treatment_ci'][0]
    ]
    ci_high = [
        control_res['control_ci'][1], 
        results[0]['treatment_ci'][1], 
        results[1]['treatment_ci'][1]
    ]
    errors = [
        [cr - ci_l for cr, ci_l in zip(conv_rates, ci_low)], 
        [ci_h - cr for cr, ci_h in zip(conv_rates, ci_high)]
    ]

    plt.figure(figsize=(10, 7))
    bars = plt.bar(labels, conv_rates, yerr=errors, capsize=7, color=VARIANT_COLORS, alpha=0.8)
    
    plt.ylabel('Conversion Rate', fontsize=12)
    plt.title('A/B Test Results: Conversion Rate by Variant (with 95% CI)', fontsize=16, pad=20)
    plt.ylim(bottom=0, top=max(ci_high) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    lift_b = (results[0]['conversion_treatment'] - control_res['conversion_control']) / control_res['conversion_control']
    lift_c = (results[1]['conversion_treatment'] - control_res['conversion_control']) / control_res['conversion_control']
    plt.text(bars[1].get_x() + bars[1].get_width()/2, bars[1].get_height() + errors[1][1] + 0.0005, 
             f'Lift: {lift_b:+.2%}', ha='center', color='#555555', fontsize=10)
    plt.text(bars[2].get_x() + bars[2].get_width()/2, bars[2].get_height() + errors[1][2] + 0.0005, 
             f'Lift: {lift_c:+.2%}', ha='center', color='#555555', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('conversion_rate_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved conversion rate plot to 'conversion_rate_comparison.png'")
    plt.show()
    
def create_plan_mix_plot(df_filtered):
    """
    Create stacked bar chart showing subscription plan mix distribution across variants.
    Visualizes the percentage of 1, 12, and 24-month plans for each test group.
    Uses consistent, light color palette for professional appearance.
    """
    print("\n9. VISUALIZING RESULTS: PLAN MIX")
    print("-" * 40)
    
    plan_mix_data = {}
    for treatment in ['A', 'B', 'C']:
        group_data = df_filtered[df_filtered['flag'] == treatment]
        counts, total = analyze_plan_mix_exclusive(group_data, f"Group {treatment}")
        if total > 0:
            plan_mix_data[treatment] = {k: v / total * 100 for k, v in counts.items()}
        else:
            plan_mix_data[treatment] = {'1_month': 0, '12_month': 0, '24_month': 0}
            
    df_plot = pd.DataFrame(plan_mix_data).T
    
    ax = df_plot.plot(kind='bar', stacked=True, figsize=(10, 7), color=PLAN_COLORS, alpha=0.8, width=0.7)
    
    plt.title('Subscription Plan Mix by Variant', fontsize=16, pad=20)
    plt.xlabel('Variant', fontsize=12)
    plt.ylabel('Percentage of Subscriptions (%)', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Plan Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', label_type='center', fontsize=9, color='#333333')
    
    plt.tight_layout()
    plt.savefig('plan_mix_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved plan mix plot to 'plan_mix_analysis.png'")
    plt.show()

def create_lift_segmentation_plot(df_filtered):
    """
    Creates bar charts showing the percentage lift of treatment variants over the control 
    for key user segments (country and device type). This is more insightful than plotting
    raw conversion rates.
    """
    print("\n10. VISUALIZING RESULTS: SEGMENTATION LIFT ANALYSIS")
    print("-" * 40)

    # Define the segments we want to analyze
    segments = {'Top Countries': 'country_group', 'Device Type': 'deviceType'}
    
    # Prepare the country_group column if it doesn't exist
    if 'country_group' not in df_filtered.columns:
        top_countries = df_filtered['country'].value_counts().head(5).index
        df_filtered['country_group'] = df_filtered['country'].apply(lambda x: x if x in top_countries else 'Other')

    # Create a figure to hold our plots
    fig, axes = plt.subplots(len(segments), 1, figsize=(12, 10))
    fig.suptitle('Percentage Lift vs. Control (Variant A) by Segment', fontsize=20, y=1.02)

    for i, (title, segment_col) in enumerate(segments.items()):
        # Calculate conversion rates per segment and variant
        segment_perf = df_filtered.groupby([segment_col, 'flag'])['has_subscription'].mean().unstack()
        
        # Calculate lift for each treatment variant relative to the control (A)
        segment_perf['lift_B'] = (segment_perf['B'] - segment_perf['A']) / segment_perf['A'] * 100
        segment_perf['lift_C'] = (segment_perf['C'] - segment_perf['A']) / segment_perf['A'] * 100
        
        # Prepare data for plotting
        lift_data = segment_perf[['lift_B', 'lift_C']].reset_index().melt(
            id_vars=segment_col, 
            var_name='Variant', 
            value_name='Lift (%)'
        )
        lift_data['Variant'] = lift_data['Variant'].str.replace('lift_', 'Variant ')

        # Create the bar plot
        sns.barplot(x='Lift (%)', y=segment_col, hue='Variant', data=lift_data, 
                    palette=[COLOR_PALETTE['variant_b'], COLOR_PALETTE['variant_c']], ax=axes[i])
        
        # Formatting
        axes[i].set_title(title, fontsize=16, pad=15)
        axes[i].axvline(0, color='black', linestyle='--')
        axes[i].set_xlabel('Lift (%) vs. Control', fontsize=12)
        axes[i].set_ylabel(segment_col.replace('_', ' ').title(), fontsize=12)
        axes[i].grid(axis='x', linestyle='--', alpha=0.6)
        
        # Add data labels
        for container in axes[i].containers:
            axes[i].bar_label(container, fmt='%.1f%%', padding=3, fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('segmentation_lift_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved segmentation lift plot to 'segmentation_lift_analysis.png'")
    plt.show()
    
    
def create_outlier_comparison_plot(df_original, df_clean):
    """
    Create comparison plot showing data before and after outlier removal.
    Uses scatter plots with log scales to visualize the impact of data cleaning.
    """
    print("\n11. VISUALIZING RESULTS: DATA CLEANING IMPACT")
    print("-" * 40)

    df = df_original.copy()
    df['pages_per_second'] = df['PageViews'] / df['sessionDurationInSeconds'].replace(0, np.nan)
    
    Q3_duration = df['sessionDurationInSeconds'].quantile(0.75)
    upper_bound_duration = Q3_duration + 3 * (Q3_duration - df['sessionDurationInSeconds'].quantile(0.25))
    
    Q3_speed = df['pages_per_second'].quantile(0.75)
    upper_bound_speed = Q3_speed + 3 * (Q3_speed - df['pages_per_second'].quantile(0.25))
    
    low_engagement_sessions = df[df['PageViews'] < 5]
    idle_threshold = low_engagement_sessions['sessionDurationInSeconds'].quantile(0.95) if len(low_engagement_sessions) > 0 else 3600
    
    duration_outliers_mask = (df['sessionDurationInSeconds'] > upper_bound_duration)
    speed_outliers_mask = (df['pages_per_second'] > upper_bound_speed) & df['pages_per_second'].notna()
    idle_outliers_mask = ((df['sessionDurationInSeconds'] > idle_threshold) & (df['PageViews'] < 5))
    
    normal_sessions_mask = ~(duration_outliers_mask | speed_outliers_mask | idle_outliers_mask)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    ax1.scatter(df.loc[normal_sessions_mask, 'sessionDurationInSeconds'], df.loc[normal_sessions_mask, 'PageViews'], 
                alpha=0.5, s=10, label='Normal Sessions')
    ax1.scatter(df.loc[idle_outliers_mask, 'sessionDurationInSeconds'], df.loc[idle_outliers_mask, 'PageViews'], 
                alpha=0.8, s=30, color='orange', label='Idle Sessions (Removed)')
    ax1.scatter(df.loc[speed_outliers_mask, 'sessionDurationInSeconds'], df.loc[speed_outliers_mask, 'PageViews'], 
                alpha=0.8, s=30, color='red', label='High Speed/Bots (Removed)')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Session Duration (seconds, log scale)', fontsize=12)
    ax1.set_ylabel('Page Views (log scale)', fontsize=12)
    ax1.set_title(f'Original Data ({len(df):,} sessions)', fontsize=16, pad=15)
    ax1.legend()
    ax1.grid(True, which="both", ls="--", alpha=0.6)

    ax2.scatter(df_clean['sessionDurationInSeconds'], df_clean['PageViews'], 
                alpha=0.5, s=10, color='green', label='Clean Sessions')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Session Duration (seconds, log scale)', fontsize=12)
    ax2.set_title(f'Clean Data ({len(df_clean):,} sessions)', fontsize=16, pad=15)
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.6)

    fig.suptitle('Visualizing the Impact of Outlier Removal', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('outlier_highlight_comparison.png', dpi=300)
    print("Saved outlier comparison plot to 'outlier_highlight_comparison.png'")
    plt.show()
    
def validate_outlier_removal(df_clean, df_original):
    """
    Analyzes the removed outlier sessions to validate that the cleaning process did not introduce bias.
    Compares distributions of key features between the clean dataset and the outlier dataset.
    """
    print("\n" + "="*50)
    print("VALIDATING OUTLIER REMOVAL FOR BIAS")
    print("="*50)

    # 1. Isolate the outlier dataframe for analysis
    clean_session_ids = df_clean['sessionId'].unique()
    df_outliers = df_original[~df_original['sessionId'].isin(clean_session_ids)].copy()
    
    if df_outliers.empty:
        print("No outliers were removed. Validation skipped.")
        return
        
    print(f"Isolated {len(df_outliers):,} outlier sessions for validation analysis.")

    # Create a 'has_subscription' column for the outliers, handling potential missing columns
    if 'subscriptionsTotal' in df_outliers.columns:
        df_outliers['subscriptionsTotal'] = pd.to_numeric(df_outliers['subscriptionsTotal'], errors='coerce').fillna(0)
        df_outliers['has_subscription'] = (df_outliers['subscriptionsTotal'] > 0).astype(int)
    else:
        print("Warning: 'subscriptionsTotal' not found in outlier data. Conversion check skipped.")
        df_outliers['has_subscription'] = 0


    # 2. Compare distribution by variant
    print("\n--- 2.1. Distribution by Variant ---")
    clean_variant_dist = df_clean['flag'].value_counts(normalize=True).rename('Clean_Data_%') * 100
    outlier_variant_dist = df_outliers['flag'].value_counts(normalize=True).rename('Outliers_%') * 100
    comparison_df_variant = pd.concat([clean_variant_dist, outlier_variant_dist], axis=1)
    print(comparison_df_variant.round(2))
    print("\n>> Assessment: Distributions should be closely matched to ensure the test remains unbiased.")

    # 3. Compare distribution by device type
    print("\n--- 2.2. Distribution by Device Type ---")
    clean_device_dist = df_clean['deviceType'].value_counts(normalize=True).rename('Clean_Data_%') * 100
    outlier_device_dist = df_outliers['deviceType'].value_counts(normalize=True).rename('Outliers_%') * 100
    comparison_df_device = pd.concat([clean_device_dist, outlier_device_dist], axis=1)
    print(comparison_df_device.round(2))
    print("\n>> Assessment: Distributions should be similar to ensure one device type was not targeted.")

    # 4. Compare distribution by Top 5 Countries (based on clean data)
    print("\n--- 2.3. Distribution by Top 5 Countries ---")
    top_countries = df_clean['country'].value_counts().head(5).index
    
    clean_geo_dist = df_clean[df_clean['country'].isin(top_countries)]['country'].value_counts(normalize=True).rename('Clean_Data_%') * 100
    outlier_geo_dist = df_outliers[df_outliers['country'].isin(top_countries)]['country'].value_counts(normalize=True).rename('Outliers_%') * 100
    comparison_df_geo = pd.concat([clean_geo_dist, outlier_geo_dist], axis=1).fillna(0)
    print(comparison_df_geo.round(2))
    print("\n>> Assessment: Proportions should be broadly similar across major markets.")

    # 5. Check the conversion rate of the outlier group
    print("\n--- 2.4. Conversion Rate of Outlier Group ---")
    outlier_conversions = df_outliers['has_subscription'].sum()
    outlier_total = len(df_outliers)
    outlier_cr = (outlier_conversions / outlier_total) * 100 if outlier_total > 0 else 0
    print(f"Total sessions in outlier group: {outlier_total:,}")
    print(f"Total conversions in outlier group: {outlier_conversions:,}")
    print(f"Outlier Conversion Rate: {outlier_cr:.4f}%")
    print("\n>> Assessment: Rate should be near-zero. A high rate suggests valid conversions were filtered out.")
    print("="*50 + "\n")
    
def create_arpu_plot(df_filtered):
    """
    Calculates and visualizes the Average Revenue Per User (ARPU) for each variant.
    Uses bootstrapping to generate robust 95% confidence intervals for the mean revenue.
    """
    print("\n11. VISUALIZING RESULTS: REVENUE ANALYSIS (ARPU)")
    print("-" * 40)

    variants = ['A', 'B', 'C']
    arpu_results = {}

    for variant in variants:
        variant_data = df_filtered[df_filtered['flag'] == variant]
        revenue_data = variant_data['subscriptionBillingsConvertedInCHFTotal'].fillna(0)

        # Calculate overall ARPU
        total_revenue = revenue_data.sum()
        total_sessions = len(variant_data)
        overall_arpu = total_revenue / total_sessions if total_sessions > 0 else 0

        # --- Bootstrapping for 95% Confidence Interval ---
        # Bootstrapping is a resampling method used to estimate a metric's distribution.
        # It's robust for non-normally distributed data like revenue.
        n_iterations = 1000
        bootstrapped_means = []
        for _ in range(n_iterations):
            # Sample with replacement from the revenue data
            sample = revenue_data.sample(n=len(revenue_data), replace=True)
            bootstrapped_means.append(sample.mean())

        # Calculate the 95% confidence interval from the bootstrapped distribution
        ci_lower = np.percentile(bootstrapped_means, 2.5)
        ci_upper = np.percentile(bootstrapped_means, 97.5)

        arpu_results[variant] = {
            'arpu': overall_arpu,
            'ci': (ci_lower, ci_upper),
            'error': [[overall_arpu - ci_lower], [ci_upper - overall_arpu]]
        }

    # --- Create the Plot ---
    plt.figure(figsize=(10, 7))

    variant_labels = [f"Variant {v}" for v in variants]
    arpu_values = [arpu_results[v]['arpu'] for v in variants]
    errors = np.array([arpu_results[v]['error'] for v in variants]).T.reshape(2, len(variants))

    bars = plt.bar(variant_labels, arpu_values, yerr=errors, capsize=7, color=VARIANT_COLORS, alpha=0.8)

    # Formatting and annotations
    plt.ylabel('ARPU (in CHF)', fontsize=12)
    plt.title('Average Revenue Per User (ARPU) by Variant (with 95% CI)', fontsize=16, pad=20)
    plt.ylim(bottom=0)
    plt.grid(axis='y', linestyle='--', alpha=0.3)

    # Add lift annotations vs. Control (A)
    control_arpu = arpu_results['A']['arpu']
    for i, bar in enumerate(bars):
        variant = variants[i]
        if variant != 'A':
            lift = (arpu_results[variant]['arpu'] - control_arpu) / control_arpu
            y_pos = bar.get_height() + errors[1, i] + (plt.ylim()[1] * 0.04) # Position above error bar
            plt.text(bar.get_x() + bar.get_width()/2, y_pos, f'Lift: {lift:+.2%}', 
                     ha='center', color='#555555', fontsize=10)

    plt.bar_label(bars, fmt='CHF %.2f', padding=3, fontsize=10)

    plt.tight_layout()
    plt.savefig('arpu_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved ARPU analysis plot to 'arpu_analysis.png'")
    plt.show()
    
def create_funnel_comparison_plot(df_clean):
    """
    Creates a side-by-side comparison of the conversion funnels for each variant.
    Visualizes the drop-off at each stage of the user journey.
    """
    print("\n12. VISUALIZING RESULTS: CONVERSION FUNNEL ANALYSIS")
    print("-" * 40)

    variants = ['A', 'B', 'C']
    stages = ['Total Sessions', 'Engaged Sessions', 'Signups', 'Subscriptions']

    funnel_data = {}

    # Calculate the numbers for each stage for each variant
    for variant in variants:
        variant_data = df_clean[df_clean['flag'] == variant]
        counts = {
            'Total Sessions': len(variant_data),
            'Engaged Sessions': variant_data['is_engaged'].sum(),
            'Signups': variant_data['has_signup'].sum(),
            'Subscriptions': variant_data['has_subscription'].sum()
        }
        funnel_data[variant] = [counts[stage] for stage in stages]

    # --- Create the Plot ---
    fig, axes = plt.subplots(1, len(variants), figsize=(18, 8), sharey=True)
    fig.suptitle('Conversion Funnel Comparison by Variant', fontsize=20)

    for i, variant in enumerate(variants):
        ax = axes[i]
        data = funnel_data[variant]
        y = np.arange(len(stages))

        # Use fill_betweenx to create the funnel shape
        # We calculate the left and right x-coordinates to center the funnel
        max_val = data[0] # Base the width on the top of the funnel
        x_left = [-(val / max_val) * 0.5 for val in data]
        x_right = [(val / max_val) * 0.5 for val in data]

        ax.fill_betweenx(y, x_left, x_right, color=VARIANT_COLORS[i], alpha=0.9)

        # Add annotations for absolute numbers and step-conversion rates
        for j, stage in enumerate(stages):
            abs_number = data[j]

            # Calculate conversion from previous step
            if j == 0:
                conv_rate_str = "100%"
            else:
                prev_step_val = data[j-1]
                conv_rate = (abs_number / prev_step_val) * 100 if prev_step_val > 0 else 0
                conv_rate_str = f"{conv_rate:.1f}%"

            # Add text in the center of the funnel
            ax.text(0, y[j], f"{stage}\n{abs_number:,}\n(Step CR: {conv_rate_str})", 
                    ha='center', va='center', fontsize=12, color='black' if i==0 else 'black')

        # Formatting
        ax.set_title(f"Variant {variant}", fontsize=16)
        ax.set_xticks([]) # Hide x-axis ticks
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('funnel_comparison_analysis.png', dpi=300, bbox_inches='tight')
    print("Saved funnel comparison plot to 'funnel_comparison_analysis.png'")
    plt.show()
    
    
print("Starting A/B test analysis for SecureNet VPN Plan Orders")
print("=" * 60)

df = load_and_validate_data()
df_clean = detect_and_remove_outliers(df)
validate_outlier_removal(df_clean, df)

df_filtered = filter_dataset_to_1_12_24_only(df_clean)

df_filtered['has_subscription'] = (df_filtered['subscriptionsTotal'] > 0).astype(int)

print("\n3. SAMPLE SIZE SUFFICIENCY CHECK")
print("-" * 40)
control_size = len(df_filtered[df_filtered['flag'] == 'A'])
treatment_b_size = len(df_filtered[df_filtered['flag'] == 'B'])
treatment_c_size = len(df_filtered[df_filtered['flag'] == 'C'])
avg_treatment_size = (treatment_b_size + treatment_c_size) / 2
run_power_analysis(control_size, avg_treatment_size)

results = run_statistical_tests(df_filtered)

analyze_segments(df_filtered)
business_impact = calculate_business_impact(df_filtered, results)

print("\n" + "="*70)
print("SUBSCRIPTION PLAN MIX ANALYSIS (1/12/24 MONTHS ONLY)")
print("="*70)

for treatment in ['A', 'B', 'C']:
    group_data = df_filtered[df_filtered['flag'] == treatment]
    analyze_plan_mix_exclusive(group_data, f"Group {treatment}")

make_recommendation(results, business_impact)
print("\nAnalysis completed successfully!")

results = run_statistical_tests(df_filtered)

create_conversion_rate_plot(results)
create_plan_mix_plot(df_filtered)
create_lift_segmentation_plot(df_filtered)
create_outlier_comparison_plot(df, df_clean)
create_arpu_plot(df_filtered)

create_funnel_comparison_plot(df_filtered)
