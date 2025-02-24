# A Google Timeline Extension for Location Classification  

## Overview  

This project extends Google Timeline’s location classification to identify significant places in a user’s daily life. It enhances Google’s Visit records using clustering algorithms and metadata analysis to improve location labeling accuracy.  

## Abstract  

Location data is crucial for research in mental health assessment and epidemic forecasting. This study develops a system to extend Google Timeline’s location classification by incorporating density-based clustering and metadata from Google APIs. A single participant's location data over 40 days was analyzed to evaluate the system’s effectiveness. The results indicate that the method provides a cost-effective and autonomous approach to location classification, offering researchers a valuable first-pass tool for analyzing significant places in an individual’s life.  

## Methodology  

### Study Design  
- Data collection spanned from **January 12, 2025, to February 21, 2025**.  
- Google Timeline data was extracted for a **single participant**.  
- Location classification was performed using clustering techniques and Google APIs.  

### Data Collection & Processing  
- **Google Timeline records**: Visit records, Activity records, and TimelinePath records were extracted.  
- **Google Place Details API** was used to retrieve location metadata.  
- **Google Places Nearby Search API** was used to discover additional significant locations.  

### Clustering Algorithm  
- A **weighted extension of DBSCAN** was used to cluster location data.  
- Clustering was based on **geolocation and time spent** at each place.  

### Labeling  
- Locations were labeled using metadata from **Google APIs**.  
- **Datamuse API** was utilized for additional contextual labeling.  

## Results  

- The clustering method successfully identified **significant locations** beyond Google’s Visit records.  
- The labeling process accurately categorized locations, though **some errors** occurred due to generic labels and misidentified nearby places.  
- A **word cloud analysis** provided insights into frequent location types.  

## Limitations  

- Clustering parameters require **fine-tuning** for different study objectives.  
- Some locations were **incorrectly labeled** due to limitations in metadata.  
- The method may **misidentify prominent nearby locations** instead of the actual places visited.  

## Future Work  

- **Refining clustering techniques** for improved accuracy.  
- **Integrating additional data sources** to enhance location classification.  
- **Expanding the study to multiple participants** for broader validation.  

## Paper  

For more details, refer to the full paper:
[**A Google Timeline Extension for Location Classification**](https://github.com/gladstone-9/cs6501-place-sense/blob/main/Place_Sense_Gabriel_Gladstone.pdf)

## Data

Source data may be provided upon request.