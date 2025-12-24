# Restaurant Menu Recommendation System Using Singular Value Decomposition (SVD)
Dining out often leads to the **"Paradox of Choice,"** where an abundance of menu options causes decision paralysis. This project implements a **Content-Based Recommendation System** that solves this problem using **Singular Value Decomposition (SVD)**.
Instead of relying on popularity ratings, this system analyzes the **"Flavor DNA"** of food. It decomposes a quantitative Item-Attribute Matrix into latent taste concepts, allowing the system to recommend dishes based on geometric similarity to a user's specific flavor preferences.

## How It Works
### 1. Matrix Decomposition ($A = U \Sigma V^T$)
The system constructs an **Item-Attribute Matrix ($A$)** where rows represent menu items and columns represent flavor attributes. SVD decomposes this matrix into:
* **$U$ (Left Singular Vectors):** Represents the coordinates of **Menu Items** in the latent concept space.
* **$\Sigma$ (Singular Values):** A diagonal matrix representing the **strength/importance** of each latent taste pattern.
* **$V^T$ (Right Singular Vectors):** Represents the coordinates of **Flavor Attributes** (Sweet, Spicy, etc.) in the latent space.

### 2. Dimensionality Reduction (Truncated SVD)
To filter out noise (minor taste variations) and capture the dominant flavor structures, we perform **Truncated SVD**. The system automatically selects the optimal rank $k$ that preserves **$\ge 95\%$ of the Cumulative Energy**, effectively reducing the complexity of the data while retaining its essence.

### 3. User Vector Projection
Since the menu items are mapped to a reduced latent space ($k$-dimensions), the user's raw preference vector ($q$) must be projected into the same space to be comparable.

$$q_{latent} = q \cdot V_k \cdot \Sigma_k^{-1}$$

### 4. Similarity Measurement
Once both the menu items and user preferences are in the same latent space, recommendations are ranked using **Cosine Similarity**:

$$Similarity = \cos(\theta) = \frac{A \cdot B}{\|A\| \|B\|}$$

## Key Features
- Quantitative representation of taste profiles using a 0–5 Likert scale

- Item–attribute matrix construction

- Automatic dimensionality reduction using Truncated SVD (≥95% variance retained)

- Latent taste pattern analysis

- Content-based recommendation using Cosine Similarity

- Interactive command-line interface

- Preset taste preference scenarios and custom user input

## Taste Attributes
Each menu item is described using the following attributes:
The system evaluates every dish based on the intensity of these 6 attributes:
| Attribute | Description |
| :--- | :--- |
| **Sweet** | Intensity of sugar, honey, or natural sweetness |
| **Salty** | Intensity of sodium-based components |
| **Sour** | Acidity levels (vinegar, citrus, fermentation) |
| **Spicy** | Heat sensation from chili or peppers |
| **Umami** | Savory profile (glutamates, meat, mushrooms) |
| **Richness** | Viscosity, creaminess, and fat content |

## Requirements
Make sure the following Python libraries are installed:
```bash
pip install numpy pandas scipy
```

## How to Run
1. Clone this github repository
```bash
https://github.com/nataliadesiany/makalahalgeo
```

2. Run the program:
```bash
python restaurant_recommendation_svd.py
```

3. Choose one of the following options:
- Enter custom taste preferences
- Use preset taste profiles

Program is ready to be used and thrilled to help you decide on what food to order!

## Author
Natalia Desiany Nursimin - 13523157
