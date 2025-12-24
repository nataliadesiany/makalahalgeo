import numpy as np
import pandas as pd
from scipy.linalg import svd, norm

# Quantitative Representation of Taste Profiles
# Taste attributes (Feature Set F)
ATTRIBUTES = ['Sweet', 'Salty', 'Sour', 'Spicy', 'Umami', 'Richness']

# Menubook dataset
# Scale 0 (Undetectable) until scale 5 (Dominant characteristics)
menu_data = [
    # DIMSUM & APPETIZER 
    {"item": "Bapao Ayam Chasiu", "scores": [4, 2, 0, 0, 3, 2], "category": "Dimsum", "price": 35200},
    {"item": "Bapao Telur Asin", "scores": [3, 3, 0, 0, 4, 4], "category": "Dimsum", "price": 35600},
    {"item": "Siomay Hong Kong Ayam", "scores": [1, 2, 0, 0, 4, 3], "category": "Dimsum", "price": 33000},
    {"item": "Lumpia Goreng Ayam", "scores": [1, 2, 0, 0, 3, 4], "category": "Dimsum", "price": 35200},
    {"item": "Kaki Ayam Tim Lada Hitam", "scores": [2, 3, 0, 3, 4, 4], "category": "Dimsum", "price": 33800},
    {"item": "Hakau Udang", "scores": [1, 2, 0, 0, 4, 1], "category": "Dimsum", "price": 35000},
    {"item": "Pangsit Udang Goreng Mayo", "scores": [3, 2, 1, 0, 4, 5], "category": "Dimsum", "price": 34900},
    {"item": "Lumpia Kulit Tahu", "scores": [1, 3, 0, 0, 5, 4], "category": "Dimsum", "price": 35200},
    {"item": "Mantau Goreng (Polos)", "scores": [2, 1, 0, 0, 1, 3], "category": "Dimsum", "price": 25000},

    # MIE, KWETIAW & NASI 
    {"item": "Mie Goreng Seafood", "scores": [2, 3, 0, 1, 5, 3], "category": "Mie", "price": 58300},
    {"item": "Mie Hong Kong Bebek Panggang", "scores": [2, 3, 0, 0, 5, 4], "category": "Mie", "price": 55000},
    {"item": "Mie Tom Yum Seafood", "scores": [1, 3, 4, 4, 4, 2], "category": "Mie", "price": 62000},
    {"item": "Kwetiau Siram Seafood", "scores": [1, 3, 0, 0, 4, 5], "category": "Mie", "price": 52800},
    {"item": "Kwetiau Goreng Sapi", "scores": [2, 4, 0, 1, 5, 4], "category": "Mie", "price": 57200},
    {"item": "Kwetiau Chili Oil", "scores": [1, 3, 1, 5, 3, 4], "category": "Mie", "price": 65000},
    {"item": "Nasi Goreng Ikan Asin Jambal", "scores": [0, 5, 0, 1, 4, 3], "category": "Nasi Goreng", "price": 52800},
    {"item": "Nasi Goreng Ayam Nanas Manis", "scores": [4, 2, 2, 0, 3, 3], "category": "Nasi Goreng", "price": 52800},
    {"item": "Nasi Goreng XO Seafood", "scores": [1, 4, 0, 3, 5, 3], "category": "Nasi Goreng", "price": 57200},
    {"item": "Nasi Goreng Yang Chow", "scores": [1, 2, 0, 0, 3, 2], "category": "Nasi Goreng", "price": 50000},
    {"item": "Nasi Putih", "scores": [1, 0, 0, 0, 1, 1], "category": "Nasi", "price": 10000},

    # SUP 
    {"item": "Sup Asam Pedas Szechuan", "scores": [1, 3, 4, 5, 4, 3], "category": "Sup", "price": 58600},
    {"item": "Sup Jagung Daging Kepiting", "scores": [3, 2, 0, 0, 4, 3], "category": "Sup", "price": 60500},
    {"item": "Sup Bibir Ikan (Fish Maw)", "scores": [1, 2, 0, 0, 3, 4], "category": "Sup", "price": 75000},
    {"item": "Sup Wonton Udang", "scores": [1, 3, 0, 0, 4, 2], "category": "Sup", "price": 45000},

    # TOFU & SAYUR
    {"item": "Mapo Tofu", "scores": [1, 4, 0, 4, 4, 3], "category": "Tofu", "price": 64900},
    {"item": "Sapo Tahu Seafood", "scores": [2, 3, 0, 0, 5, 4], "category": "Tofu", "price": 72600},
    {"item": "Tahu Sutra Telur Asin", "scores": [1, 3, 0, 0, 4, 5], "category": "Tofu", "price": 53900},
    {"item": "Tahu Goreng Lada Garam", "scores": [0, 4, 0, 3, 3, 2], "category": "Tofu", "price": 48000},
    {"item": "Brokoli Cah Bawang Putih", "scores": [1, 2, 0, 0, 3, 1], "category": "Sayur", "price": 45000},
    {"item": "Baby Buncis Cah Sapi", "scores": [2, 4, 0, 1, 4, 3], "category": "Sayur", "price": 52000},
    {"item": "Kailan Dua Rasa", "scores": [1, 3, 0, 0, 3, 2], "category": "Sayur", "price": 50000},
    {"item": "Cap Cay Goreng", "scores": [2, 3, 0, 0, 4, 3], "category": "Sayur", "price": 48000},

    # AYAM & BEBEK
    {"item": "Ayam Goreng Asam Manis", "scores": [5, 2, 4, 0, 3, 3], "category": "Ayam", "price": 61600},
    {"item": "Ayam Kung Pao", "scores": [3, 3, 1, 3, 4, 3], "category": "Ayam", "price": 61600},
    {"item": "Ayam Saus Lemon", "scores": [4, 1, 5, 0, 2, 2], "category": "Ayam", "price": 61600},
    {"item": "Ayam Goreng Mentega", "scores": [4, 4, 1, 0, 4, 5], "category": "Ayam", "price": 62000},
    {"item": "Ayam Rebus Pek Cam Kee", "scores": [1, 4, 0, 0, 3, 3], "category": "Ayam", "price": 60000},
    {"item": "Bebek Panggang (1/4)", "scores": [3, 4, 0, 0, 5, 5], "category": "Bebek", "price": 95000},

    # SAPI 
    {"item": "Sapi Lada Hitam", "scores": [2, 4, 0, 4, 5, 4], "category": "Sapi", "price": 79600},
    {"item": "Sapi Mongolia", "scores": [3, 4, 0, 2, 4, 4], "category": "Sapi", "price": 77000},
    {"item": "Sapi Cah Kailan", "scores": [1, 3, 0, 0, 4, 3], "category": "Sapi", "price": 75000},
    {"item": "Sapi Szechuan (Mala)", "scores": [1, 4, 1, 5, 5, 5], "category": "Sapi", "price": 82000},

    # SEAFOOD 
    {"item": "Cumi Goreng Lada Garam", "scores": [0, 5, 0, 3, 4, 2], "category": "Seafood", "price": 68200},
    {"item": "Cumi Goreng Tepung", "scores": [1, 3, 0, 0, 3, 3], "category": "Seafood", "price": 65000},
    {"item": "Udang Telur Asin", "scores": [2, 4, 0, 0, 5, 5], "category": "Seafood", "price": 80300},
    {"item": "Udang Mayonnaise", "scores": [4, 2, 1, 0, 4, 5], "category": "Seafood", "price": 78000},
    {"item": "Ikan Gurame Saus Asam Manis", "scores": [5, 2, 4, 0, 3, 3], "category": "Seafood", "price": 111100},
    {"item": "Ikan Gurame Tahu Tausi", "scores": [2, 4, 0, 0, 5, 3], "category": "Seafood", "price": 105000},
    {"item": "Ikan Dori Tim Nyonya", "scores": [2, 3, 5, 3, 4, 2], "category": "Seafood", "price": 70000},
    {"item": "Fu Yung Hai (Kepiting)", "scores": [5, 2, 3, 0, 4, 4], "category": "Seafood", "price": 65000},

    # BUBUR 
    {"item": "Bubur Polos", "scores": [1, 1, 0, 0, 2, 2], "category": "Bubur", "price": 29700},
    {"item": "Bubur 3 Topping", "scores": [1, 3, 0, 0, 4, 3], "category": "Bubur", "price": 41800},
    {"item": "Bubur Ikan", "scores": [1, 2, 0, 0, 4, 2], "category": "Bubur", "price": 35000},
]

# Dataframe conversion
df_menu = pd.DataFrame(menu_data)
df_vector = pd.DataFrame(df_menu['scores'].tolist(), columns=ATTRIBUTES, index=df_menu['item'])

print("="*80)
print("RESTAURANT MENU RECOMMENDATION SYSTEM USING SVD")
print("="*80)
print(f"Total Menu Items: {len(df_menu)}")
print(f"Matrix Shape: {df_vector.shape} (Items x Attributes)")

# Pattern Extraction (SVD)
class SVDRecommender:
    def __init__(self, data_matrix):
        self.raw_matrix = data_matrix.values
        self.items = data_matrix.index
        self.attributes = data_matrix.columns
        self.U = None
        self.Sigma = None
        self.Vt = None
        self.k = None
        self.Vk = None
        self.Sigma_k_inv = None
        self.Uk_Sigmak = None
        
    def fit(self, energy_threshold=0.95):
        # Matrix Factorization
        U, s, Vt = svd(self.raw_matrix, full_matrices=False)
        
        # Cumulative Energy Calculation
        energy_sq = s ** 2
        total_energy = np.sum(energy_sq)
        cumulative_energy = np.cumsum(energy_sq) / total_energy
        
        # Determine optimal k
        self.k = np.argmax(cumulative_energy >= energy_threshold) + 1
        
        print(f"\n{'='*80}")
        print("SVD ANALYSIS")
        print("="*80)
        print(f"Singular Values: {np.round(s, 2)}")
        print(f"Cumulative Energy: {np.round(cumulative_energy * 100, 1)}%")
        print(f"Optimal k selected: {self.k} (Retains {cumulative_energy[self.k-1]*100:.1f}% variance)")
        
        # Truncate
        self.U = U[:, :self.k]
        self.Sigma = np.diag(s[:self.k])
        self.Vt = Vt[:self.k, :]
        
        # Components for projection
        self.Vk = self.Vt.T
        self.Sigma_k_inv = np.linalg.inv(self.Sigma)
        self.Uk_Sigmak = np.dot(self.U, self.Sigma)
        
    def explain_latent_concepts(self):
        concepts = pd.DataFrame(self.Vt, columns=self.attributes)
        concepts.index = [f"Latent Concept {i+1}" for i in range(self.k)]
        return concepts

# Train Model
print("\n" + "="*80)
print("TRAINING MODEL...")
print("="*80)
recommender = SVDRecommender(df_vector)
recommender.fit()

# Display taste patterns
print("\n" + "="*80)
print("LATENT TASTE CONCEPTS")
print("="*80)
print(recommender.explain_latent_concepts())

# Recommendation algorithm
def get_cosine_similarity(vec1, vec2):
    norm1 = norm(vec1)
    norm2 = norm(vec2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)

def recommend_menu(user_input_vector, model, top_n=5):
    q_raw = np.array(user_input_vector)
    q_latent = np.dot(np.dot(q_raw, model.Vk), model.Sigma_k_inv)
    
    results = []
    for idx, item_name in enumerate(model.items):
        m_latent = model.Uk_Sigmak[idx]
        sim_score = get_cosine_similarity(q_latent, m_latent)
        
        meta = df_menu.loc[df_menu['item'] == item_name].iloc[0]
        results.append({
            "Menu Item": item_name,
            "Similarity": sim_score,
            "Category": meta['category'],
            "Price": f"Rp {meta['price']:,}",
            "Flavor Profile": df_vector.loc[item_name].values
        })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="Similarity", ascending=False).head(top_n)
    return results_df, q_latent

def get_user_input():
    print("Enter your taste preferences")
    print("="*80)
    print("Scale: 0 = Absent, 1 = Trace, 3 = Moderate, 5 = Dominant")
    print("-"*80)
    
    preferences = []
    
    for attr in ATTRIBUTES:
        while True:
            try:
                value = int(input(f"  {attr:<12} (0-5): "))
                if 0 <= value <= 5:
                    preferences.append(value)
                    break
                else:
                    print("Enter a number between 0-5!")
            except ValueError:
                print("Enter a valid number!")
    
    return preferences

def display_recommendations(recs):
    print("Your Personalized Recommendations")
    print("="*80)
    
    for i, (idx, row) in enumerate(recs.iterrows(), 1):
        print(f"\n{i}. {row['Menu Item']}")
        print(f"   Category   : {row['Category']}")
        print(f"   Price      : {row['Price']}")
        print(f"   Similarity : {row['Similarity']:.4f}")
        print(f"   Taste      : Sweet={row['Flavor Profile'][0]}, Salty={row['Flavor Profile'][1]}, "
              f"Sour={row['Flavor Profile'][2]}, Spicy={row['Flavor Profile'][3]}, "
              f"Umami={row['Flavor Profile'][4]}, Rich={row['Flavor Profile'][5]}")

def show_preset_examples():
    print("\n" + "="*80)
    print("PRESET EXAMPLES")
    print("="*80)
    print("1. Sweet & Sour Lover    : [5, 2, 5, 0, 3, 2]")
    print("2. Spicy & Umami Lover   : [1, 3, 0, 5, 5, 3]")
    print("3. Salty & Rich Lover    : [1, 5, 0, 0, 4, 5]")
    print("4. Balanced Taste        : [3, 3, 3, 3, 3, 3]")
    print("5. Mild & Light          : [1, 1, 0, 0, 2, 1]")

def main():
    print("\n" + "="*80)
    print("Ready to show recommendations!")
    print("="*80)
    
    while True:
        print("\n" + "="*80)
        print("Menu Options")
        print("="*80)
        print("1. Enter your own taste preferences (custom)")
        print("2. Use preset example")
        print("3. Exit")
        print("-"*80)
        
        choice = input("Choose option (1/2/3): ").strip()
        
        if choice == '1':
            # Custom input
            user_pref = get_user_input()
            
            print("\n Your preferences: ", end="")
            for i, attr in enumerate(ATTRIBUTES):
                print(f"{attr}={user_pref[i]}", end=" ")
            print()
            
            # Get number of recommendations
            while True:
                try:
                    top_n = int(input("\nHow many recommendations do you want? (1-20): "))
                    if 1 <= top_n <= 20:
                        break
                    else:
                        print("Masukkan angka antara 1-20!")
                except ValueError:
                    print("Masukkan angka yang valid!")
            
            # Generate recommendations
            recs, _ = recommend_menu(user_pref, recommender, top_n=top_n)
            display_recommendations(recs)
            
        elif choice == '2':
            # Preset examples
            show_preset_examples()
            
            preset_choice = input("\nChoose preset (1-5): ").strip()
            
            presets = {
                '1': ([5, 2, 5, 0, 3, 2], "Sweet & Sour Lover"),
                '2': ([1, 3, 0, 5, 5, 3], "Spicy & Umami Lover"),
                '3': ([1, 5, 0, 0, 4, 5], "Salty & Rich Lover"),
                '4': ([3, 3, 3, 3, 3, 3], "Balanced Taste"),
                '5': ([1, 1, 0, 0, 2, 1], "Mild & Light"),
            }
            
            if preset_choice in presets:
                user_pref, desc = presets[preset_choice]
                print(f"\n Selected: {desc}")
                print(f"Preferences: {user_pref}")
                
                # Generate recommendations
                recs, _ = recommend_menu(user_pref, recommender, top_n=10)
                display_recommendations(recs)
            else:
                print("Invalid preset choice!")
                
        elif choice == '3':
            print("\n" + "="*80)
            print("Thank you for using the SVD Restaurant Menu Recommendation System!")
            print("="*80)
            break
        else:
            print("Invalid choice! Please select 1, 2, or 3.")
        
        cont = input("\n\nWant to try another recommendation? (y/n): ").strip().lower()
        if cont != 'y':
            print("\n" + "="*80)
            print("Thank you for using the SVD Restaurant Menu Recommendation System!")
            print("="*80)
            break

if __name__ == "__main__":
    main()