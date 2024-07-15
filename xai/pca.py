import numpy as np
from PIL import Image

import plotly.express as px

from glob import iglob
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


from scipy import stats
from scipy.stats import zscore
from scipy.stats import gaussian_kde
from scipy.stats import entropy
np.random.seed(42)

class PCAAnalysis:
    def __init__(self, real_images, fake_images, n_components=10):
        self.load_datasources(real_images, fake_images)
        self.n_components = n_components

    def load_datasources(self, real_images, fake_images):
        self.real_images_list = list(iglob(real_images))
        self.fake_images_list = list(iglob(fake_images))
        self.num_images = len(self.real_images_list)

        self.fake_images = np.random.choice(self.fake_images_list, self.num_images)
        self.labels =  ["Real"] * self.num_images + ["Synthetic"] * self.num_images

        self.flattened_real = []
        self.flattened_fake = []
        for i in range(self.num_images):
            real = Image.open(self.real_images_list[i]).convert("L")
            fake = Image.open(self.fake_images_list[i]).convert("L")
            if real.size != (224, 224):
                real = real.resize((224, 224))
            if fake.size != (224, 224):
                fake = fake.resize((224, 224))
            real = np.array(real).flatten()
            fake = np.array(fake).flatten()
            self.flattened_real.append(real)
            self.flattened_fake.append(fake)
        
        self.real_images, self.synthetic_images = np.array(self.flattened_real), np.array(self.flattened_fake)
        
        self.real_images, self.synthetic_images = self.real_images - np.average(self.real_images, axis=0), self.synthetic_images - np.average(self.synthetic_images, axis=0)
        self.combined = np.vstack([ self.real_images, self.synthetic_images])
        standard_scaler = StandardScaler()

        self.combined = standard_scaler.fit_transform(self.combined)
        self.real_images = self.combined[:self.num_images]
        self.synthetic_images = self.combined[self.num_images:]

    def refit_without_outliers(self, outlier_list):
        self.real_images = [self.real_images[i] for i in range(len(self.real_images)) if i not in outlier_list]
        # self.synthetic_images =  [self.synthetic_images[i] for i in range(len(self.synthetic_images)) if i not in outlier_list[1]]
        self.labels = ["Real"] * len(self.real_images) + ["Synthetic"] * len(self.synthetic_images)

        self.combined = np.vstack([self.real_images, self.synthetic_images])
        self.fit()

    def fit(self):
        self.pca = PCA(n_components=self.n_components, random_state=42)
        self.pca.fit(self.real_images)
        self.real_scores = self.pca.transform(self.real_images)

        # Project synthetic features onto the principal components obtained from real images
        self.synthetic_scores = self.pca.transform(self.synthetic_images)


    def visualize_components(self):
        print(len(self.real_scores))
        stacked = np.concatenate([self.real_scores, self.synthetic_scores])
        fig = px.scatter(stacked, x=0, y=1, color=self.labels)
        fig.show()

    def pc_score_analysis(self, first_n_components=2):
        # Calculate statistics for PCA score distributions
        for i in range(first_n_components):
            # Extract scores for current principal component
            real_scores_pc = self.real_scores[:, i]
            synthetic_scores_pc = self.synthetic_scores[:, i]
          
            # Calculate KL divergence
            # Use gaussian_kde to estimate the density functions
            kde_real = gaussian_kde(real_scores_pc)
            kde_synthetic = gaussian_kde(synthetic_scores_pc)
            
            # Define the range for evaluation
            xmin = min(np.min(real_scores_pc), np.min(synthetic_scores_pc))
            xmax = max(np.max(real_scores_pc), np.max(synthetic_scores_pc))
            x_vals = np.linspace(xmin, xmax, 1000)
            
            # Calculate densities
            density_real = kde_real(x_vals)
            density_synthetic = kde_synthetic(x_vals)
            
            # Calculate KL divergence between the distributions
            kl_divergence = entropy(density_real, density_synthetic)
            
            # Print results for the current principal component
            print(f"Principal Component {i+1}:")
            print(f"KL Divergence: {kl_divergence:.4f}")
            print()

    def detect_outliers(self, first_n_components=2):    
        # Perform outlier detection using Isolation Forest
        outlier_detector = IsolationForest(contamination="auto",  random_state=42)
        outlier_detector.fit(np.vstack([self.real_scores[:, :first_n_components], self.synthetic_scores[:, :first_n_components]]))
        outliers_real = outlier_detector.predict(self.real_scores[:, :first_n_components])
        outliers_synthetic = outlier_detector.predict(self.synthetic_scores[:, :first_n_components])
        
        # Count the number of outliers detected
        num_outliers_real = np.sum(outliers_real == -1)
        num_outliers_synthetic = np.sum(outliers_synthetic == -1)

        real_outliers = np.where(outliers_real == -1)
        synthetic_outliers = np.where(outliers_synthetic == -1)
        # self.real_scores = self.real_scores[[idx for idx in range(0, len(self.real_scores)) if idx not in real_outliers[0]], :]
        # self.synthetic_scores = self.synthetic_scores[[idx for idx in range(0, len(self.synthetic_scores)) if idx not in synthetic_outliers[0]], :]

        print(f"Outlier Analysis Results:")
        print(f"  Number of outliers in real images: {num_outliers_real}")
        print(f"  Number of outliers in synthetic images: {num_outliers_synthetic}")
        return real_outliers[0], synthetic_outliers[0]

    def statistical_significance_test(self, first_n_components=2):
        # Perform t-tests for each principal component
        for i in range(first_n_components):
            real_scores_pc = self.real_scores[:, i]
            synthetic_scores_pc = self.synthetic_scores[:, i]
            
            real_stat, real_p_value = stats.shapiro(real_scores_pc)
            synthetic_stat, synthetic_p_value = stats.shapiro(synthetic_scores_pc)

            print(f'Shapiro-Wilks for Real PCA Prinicpal Component {i}: Statistic={real_stat}, p-value={real_p_value}')
            print(f'Shapiro-Wilks Synthetic PCA Prinicpal Component {i}: Statistic={synthetic_stat}, p-value={synthetic_p_value}')
            alpha = 0.05
            if real_p_value > alpha and synthetic_p_value > alpha:
                print('Both samples look Gaussian (fail to reject H0)')
                print()
                # Perform t-test
                t_stat, t_p_value = stats.ttest_ind(real_scores_pc, synthetic_scores_pc)
                print(f'T-Test: Statistic={t_stat}, p-value={t_p_value}')
            else:
                print('One or both samples do not look Gaussian (reject H0)')
                print()
                # Perform Mann-Whitney U Test
                u_stat, u_p_value = stats.mannwhitneyu(real_scores_pc, synthetic_scores_pc)
                print(f'Mann-Whitney U Test: Statistic={u_stat}, p-value={u_p_value}')
            print()


