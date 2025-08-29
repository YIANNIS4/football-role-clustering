#  football-role-clustering

**MSc Football Data Analysis project**: Role-based player profiling using PCA and K-Means clustering, with interactive Streamlit dashboards for scouting and tactical insights.

##  Tactical Role Profiling Dashboard

An interactive Streamlit application developed as part of a master's thesis in Sports Analytics. The dashboard enables tactical exploration of player roles and performance metrics across top football leagues, supporting scouting workflows, academic transparency, and data-driven decision-making.

##  Live App

Access the deployed dashboard here: [Streamlit App](https://football-role-clustering-ypguqyzfku3zq7vkatqgef.streamlit.app)


## Repository Contents

- `main_streamlit_app.py`: Main Streamlit script powering the dashboard  
- `requirements.txt`: Python dependencies required for deployment  
- `README.md`: Overview of the project, features, and structure  
- `notebooks/`: Jupyter notebooks for league-specific and positional analysis  
- `data/`: CSV datasets used in the app (each under 100MB)

>  Note: All notebooks listed below are referenced in the thesis annex section and are submitted in a separate folder as per university guidelines.  
> PDF exports of notebooks and scripts are included in the university submission folder but not hosted here due to GitHub file size limits.

##  Thesis Document

The full thesis document (`Thesis_Document.docx`) is included in the university submission folder. It provides detailed context, methodology, and critical discussion of the results. All code and dashboards in this repository support the thesis findings and are referenced throughout the document.

##  Notebook Index

Each notebook corresponds to a specific league and positional analysis:

| Notebook                 | Purpose |

| `Global_defenders.ipynb` | Clustering and profiling of global defender data |
| `Global_midfielders.ipynb` | Clustering and profiling of global midfielder data |
| `Global_forwards.ipynb` | Clustering and profiling of global forward data |
| `LA_LIGA_FORWARDS.ipynb` | Tactical profiling of La Liga forwards |
| `La_Liga_midfielders_defenders.ipynb` | Combined analysis of La Liga midfielders and defenders |
| `Ligue1_defenders.ipynb` | Tactical profiling of Ligue 1 defenders |
| `Ligue1_midfielders.ipynb` | Tactical profiling of Ligue 1 midfielders |
| `Ligue1_forwards.ipynb` | Tactical profiling of Ligue 1 forwards |
| `SERIE_A_midfielders.ipynb` | Tactical profiling of Serie A midfielders |
| `SERIE_A_forwards.ipynb` | Tactical profiling of Serie A forwards |
| `SERIE_A_defenders.py` | Script-based analysis of Serie A defenders |
| `streamlit_defenders_app.py` | Streamlit logic for defender dashboard |
| `streamlit_midfielders_app.py` | Streamlit logic for midfielder dashboard |
| `streamlit.txt` | Notes or setup instructions for Streamlit environment |

##  App Features

- **Positional Dashboard Selection**: Choose between Defenders, Midfielders, and Forwards, each with tailored metrics and tactical filters  
- **Tactical Role Filters**: Explore players grouped by role (e.g., Ball-Winning Midfielder, Wide Forward)  
- **Player Comparison Tools**: Use dropdowns to compare players via radar charts and bar charts based on selected metrics  
- **Individual Key Performance Areas**: PCA-derived components summarizing player tendencies  
- **Metric Type Toggle**: Switch between raw metrics, PCA components, and grouped KPAs (especially useful for defenders)  
- **Custom Metric Selection**: Choose specific metrics to visualize in radar and bar charts  
- **Player Table Filtering**: View raw performance metrics and isolate individual players  
- **Explanatory Section**: Expandable section detailing tactical roles and performance dimensions for interpretability

##  Run Locally 

To run the app locally on your machine:

1. **Clone the repository**  
   This downloads the project files to your computer: git clone https://github.com/YIANNIS4/football-role-clustering.git

2. **Install dependencies**  
This installs all required Python packages listed in `requirements.txt`:  pip install -r requirements.txt


3. **Launch the app**  
This starts the Streamlit server and opens the dashboard in your browser: streamlit run main_streamlit_app.py


>  If you're using a virtual environment (e.g. `venv`), make sure it's activated before running these commands.

##  Author

**Ioannis Kastritis** — MSc candidate in Football Data Analytics  
Focused on bridging technical rigor with tactical football insight.

##  Acknowledgments

This project was developed as part of a thesis submission for **Sports Data Campus**, Universidad Católica San Antonio de Murcia. All data used complies with university guidelines and is structured for examiner accessibility.

##  Limitations and Future Work

While the project successfully delivered a functional dashboard and tactical clustering framework, several limitations remain:

- **Data limitations**: The event datasets lacked consistency in advanced contextual metrics such as pressures, progressive actions, and passing ranges. This constrained the granularity of role definitions and limited the depth of tactical interpretation.

- **Methodological limitations**: The use of K-Means clustering assumes spherical clusters with equal variance, which does not fully capture the overlapping and hybrid nature of football roles. For example, Aurélien Tchouaméni was misclassified as a playmaker due to these constraints, highlighting the need for more flexible clustering approaches.

- **Application limitations**: The analysis focused on three leagues (La Liga, Ligue 1, Serie A), which offered stylistic diversity but may not generalize to other contexts such as the Premier League or international tournaments. Additionally, the model treated roles as static, whereas in practice they shift dynamically across phases of play.

- **Technical implementation limitations**: A small number of players displayed zero values in specific metrics, typically due to limited match involvement or gaps in event data coverage. These cases were retained to preserve dataset completeness but may underrepresent certain performance dimensions. They underscore the importance of contextual interpretation when comparing player profiles.

###  Future Work

- Incorporate more dynamic role modeling techniques that account for phase-of-play transitions and hybrid responsibilities  
- Expand the dataset to include additional leagues and competitions, improving generalizability and tactical diversity  
- Explore alternative clustering algorithms (e.g. DBSCAN, Gaussian Mixture Models) to better capture role fluidity and overlapping player profiles  
- Integrate richer event data with contextual tags (e.g. pressure zones, pass types, off-ball movements) to refine role definitions and enhance scouting utility





