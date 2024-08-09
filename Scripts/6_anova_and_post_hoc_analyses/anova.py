import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.power import TTestIndPower
import os


"""
Set paths to datasets
"""
root_datasets = "C:/Users/HP/PycharmProjects/MPH/1data/anova_sec_data/"
path_accuracy_data = root_datasets + "1_accuracy.xlsx"
path_precision_data = root_datasets + "2_precision.xlsx"
path_f1_score_data = root_datasets + "3_f1_score.xlsx"
path_specificity_data = root_datasets + "4_specificity.xlsx"
path_npv_data = root_datasets + "5_npv.xlsx"
path_cohens_kappa_data = root_datasets + "6_cohens_kappa.xlsx"
path_roc_auc_data = root_datasets + "7_roc_auc.xlsx"

"""
Store paths to datasets in a dictionary
"""
dictionary_datasets_paths = {
    'accuracy': path_accuracy_data,
    'precision': path_precision_data,
    'f1_score': path_f1_score_data,
    'specificity': path_specificity_data,
    'npv': path_npv_data,
    'cohens_kappa': path_cohens_kappa_data,
    'roc_auc': path_roc_auc_data, }


"""
Set path to results
"""
root_results = "C:/Users/HP/PycharmProjects/MPH/4results/6_anova/"
path_results_accuracy = root_results + "1_accuracy/"
path_results_precision = root_results + "2_precision/"
path_results_f1_score = root_results + "3_f1_score/"
path_results_specificity = root_results + "4_specificity/"
path_results_npv = root_results + "5_npv/"
path_results_cohens_kappa = root_results + "6_cohens_kappa/"
path_results_roc_auc = root_results + "7_roc_auc/"

"""
Store paths to results in a dictionary
"""
dictionary_results_paths = {
    'accuracy': path_results_accuracy,
    'precision': path_results_precision,
    'f1_score': path_results_f1_score,
    'specificity': path_results_specificity,
    'npv': path_results_npv,
    'cohens_kappa': path_results_cohens_kappa,
    'roc_auc': path_results_roc_auc, }


"""
Perform ANOVA and post hoc analyses on all datasets
"""


def perform_anova_and_post_hoc_analyses():

    for dataset_name, dataset_path in dictionary_datasets_paths.items():
        """
        Load dataset 
        """
        raw_data = pd.read_excel(dataset_path, engine='openpyxl')

        """
        Melt the dataframe for easier analysis
        """
        melted_data = pd.melt(
            raw_data, id_vars=['Technique', 'Model'],
            value_vars=['D(1)', 'D(2)', 'D(3)', 'D(4)'],
            var_name='Dataset', value_name=dataset_name)

        """
        Two-way ANOVA with interaction
        """
        model = ols(
            f'{dataset_name} ~ C(Technique) * C(Model)',
            data=melted_data).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        anova_table.to_csv(os.path.join(
            dictionary_results_paths[dataset_name],
            f'1_two_way_ANOVA_results_for_{str.upper(dataset_name)}_.csv'))

        """
        Plotting interaction plot
        """
        sns.pointplot(
            x='Technique', y=dataset_name, hue='Model',
            data=melted_data, dodge=True, markers=['o', 's'], capsize=.1)
        plt.title(f'Interaction Plot for {str.upper(dataset_name)} metric')
        plt.savefig(os.path.join(
            dictionary_results_paths[dataset_name],
            f'_interaction_plot_for_{str.upper(dataset_name)}_.png'))
        plt.clf()

        """
        Tukey's HSD test
        """
        tukey = pairwise_tukeyhsd(
            endog=melted_data[dataset_name],
            groups=melted_data['Technique'] + melted_data['Model'], alpha=0.05)
        tukey_summary = tukey.summary()

        file_path_tukey = os.path.join(
                dictionary_results_paths[dataset_name],
                f'2_tukey_hsd_results_for_{str.upper(dataset_name)}_.txt')

        if not os.path.exists(file_path_tukey):
            with open(file_path_tukey, 'w') as f:
                f.write(str(tukey_summary))

        """
        Bonferroni correction
        """
        p_values = anova_table['PR(>F)'].values
        reject, p_values_corrected, _, _ = multipletests(
            p_values, method='bonferroni')
        bonferroni_results = pd.DataFrame({
            'p_values': p_values, 'p_values_corrected': p_values_corrected,
            'reject': reject})
        bonferroni_results.to_csv(os.path.join(
            dictionary_results_paths[dataset_name],
            f'3_bonferroni_results_for_{str.upper(dataset_name)}_.csv'))

        """
        Holm-Bonferroni method
        """
        reject, p_values_corrected, _, _ = multipletests(
            p_values, method='holm')
        holm_bonferroni_results = pd.DataFrame({
            'p_values': p_values, 'p_values_corrected': p_values_corrected,
            'reject': reject})
        holm_bonferroni_results.to_csv(os.path.join(
            dictionary_results_paths[dataset_name],
            f'4_holms_bonferroni_results_for_{str.upper(dataset_name)}_.csv'))

        """
        Power Analysis
        """
        effect_size = model.rsquared
        analysis = TTestIndPower()
        power = analysis.solve_power(
            effect_size=effect_size, nobs1=len(melted_data), alpha=0.05)

        file_path_power = os.path.join(
                dictionary_results_paths[dataset_name],
                f'5_power_analysis_for_{str.upper(dataset_name)}_.txt')

        if not os.path.exists(file_path_power):
            with open(file_path_power, 'w') as f:
                f.write(f'Effect Size: {effect_size}\nPower: {power}')

        """
        Calculate effect sizes
        """
        eta_squared = anova_table['sum_sq'][:-1] / anova_table['sum_sq'].sum()
        partial_eta_squared = anova_table['sum_sq'][:-1] / (anova_table['sum_sq'][:-1] + anova_table['sum_sq'][-1])
        effect_sizes = pd.DataFrame({'eta_squared': eta_squared, 'partial_eta_squared': partial_eta_squared})
        effect_sizes.to_csv(os.path.join(
                dictionary_results_paths[dataset_name],
                f'6_effect_sizes_for_{str.upper(dataset_name)}_.csv'))

        """
        Sensitivity analyses
        """
        sensitivity_results = list()
        for alpha in [0.01, 0.05, 0.1]:
            reject, p_values_corrected, _, _ = multipletests(
                p_values, alpha=alpha, method='holm')
            sensitivity_results.append({
                'alpha': alpha, 'reject': reject,
                'p_values_corrected': p_values_corrected})
        sensitivity_df = pd.DataFrame(sensitivity_results)
        sensitivity_df.to_csv(os.path.join(
                dictionary_results_paths[dataset_name],
                f'7_sensitivity_analyses_for_{str.upper(dataset_name)}_.csv'))

        """ 
        Bar plots
        """
        sns.barplot(x='Technique', y=dataset_name, hue='Model',
                    data=melted_data, ci='sd')
        plt.title(f'Mean {str.upper(dataset_name)} by Technique and Model')
        plt.savefig(os.path.join(
            dictionary_results_paths[dataset_name],
            f'_bar_plot_for_{str.upper(dataset_name)}_.png'))
        plt.clf()

        # # Heatmaps
        # heatmap_data = melted_data.pivot(
        #     index='Technique', columns='Model', values=dataset_name)
        # sns.heatmap(heatmap_data, annot=True, cmap='coolwarm')
        # plt.title(f'Heatmap of {str.upper(dataset_name)}')
        # plt.savefig(os.path.join(
        #     dictionary_results_paths[dataset_name],
        #     f'_heatmap_for_{str.upper(dataset_name)}_.png'))
        # plt.clf()


perform_anova_and_post_hoc_analyses()
