import React, { useState } from 'react'
import * as tf from '@tensorflow/tfjs';
import * as tfvis from '@tensorflow/tfjs-vis';
import { saveAs } from 'file-saver';

const App = () => {

  const [value, setValue] = useState(null);
  const [loading, setLoading] = useState(false);
  const [trainingData, setTrainingData] = useState([]);

  const [longit, setLongit] = useState('');
  const [latit, setLatit] = useState('');
  const [hma, setHma] = useState('');
  const [tr, setTr] = useState('');
  const [tb, setTb] = useState('');
  const [pop, setPop] = useState('');
  const [hHold, setHHold] = useState('');
  const [minc, setMinc] = useState('');
  const [prox, setProx] = useState('');





  async function getData() {
    const houseDataResponse = await fetch('http://127.0.0.1:5173/houseData.json');
    const houseData = await houseDataResponse.json();
    const cleaned = houseData.map(house => {
      let proximity;
      switch (house.ocean_proximity) {
        case "NEAR BAY":
          proximity = Proximity.NEAR_BAY;
          break;
        case "<1H OCEAN":
          proximity = Proximity.CLOSE_TO_OCEAN;
          break;
        case "NEAR OCEAN":
          proximity = Proximity.NEAR_OCEAN;
          break;
        case "INLAND":
          proximity = Proximity.INLAND;
          break;
        default:
          proximity = null;
      }
      
      const longitude = house.longitude ? parseFloat(house.longitude) : null;
      const latitude = house.latitude ? parseFloat(house.latitude) : null;
      const housing_median_age = house.housing_median_age ? parseFloat(house.housing_median_age) : null;
      const total_rooms = house.total_rooms ? parseFloat(house.total_rooms) : null;
      const total_bedrooms = house.total_bedrooms ? parseFloat(house.total_bedrooms) : null;
      const population = house.population ? parseFloat(house.population) : null;
      const households =house.households ? parseFloat(house.households) : null;
      const median_income = house.median_income ?  parseFloat(house.median_income) : null;
      const median_house_value = house.median_house_value ?  parseFloat(house.median_house_value) : null;
  
      if (isNaN(longitude) || isNaN(latitude) || isNaN(housing_median_age) || isNaN(total_rooms) || isNaN(population) || isNaN(households) || isNaN(median_income) || isNaN(total_bedrooms)  || isNaN(median_house_value) || proximity === null || longitude == null || latitude == null || housing_median_age == null || total_rooms == null || total_bedrooms == null || population == null || households == null || median_income == null || median_house_value == null) {
        return null;
      }
  
      return {
        longitude,
        latitude,
        housing_median_age,
        total_rooms,
        total_bedrooms,
        population,
        households,
        median_income,
        median_house_value,
        proximity,
      };
    }).filter(house => house !== null);
  
    return cleaned;
  }
  
  const Proximity = {
    NEAR_BAY: 1,
    CLOSE_TO_OCEAN: -1,
    NEAR_OCEAN: 0,
    INLAND: 2,
  };

  function convertToTensor(data) {

    return tf.tidy(() => {
      // Step 1. Brasser les données
      tf.util.shuffle(data);

      // Step 2. Traiter et Convertir les données en tensors

      const inputs = data.map(d => [
        d.longitude,
        d.latitude,
        d.housing_median_age,
        d.total_rooms,
        d.total_bedrooms,
        d.population,
        d.households,
        d.median_income,
        d.proximity,
      ])
      const labels = data.map(d => d.median_house_value);


      const inputTensor = tf.tensor2d(inputs, [inputs.length, 9]);
      const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

      //Step 3. Normaliser les données dans une marge 0 - 1 en utilisant le 'min-max scaling'
      const inputMax = inputTensor.max();
      const inputMin = inputTensor.min();
      const labelMax = labelTensor.max();
      const labelMin = labelTensor.min();

      const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
      const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));

      return {
        inputs: normalizedInputs,
        labels: normalizedLabels,

        inputMax,
        inputMin,
        labelMax,
        labelMin,
      }
    });
  }

  async function run() {
    // Charger et afficher les données d'entrainement
    const data = await getData();

    console.log(data);

    // Créer le model
    const model = createModel();
    tfvis.show.modelSummary({ name: 'Model Summary' }, model);

    // Recupérer les données pour l'entrainement
    const tensorData = convertToTensor(data);
    const { inputs, labels } = tensorData;

    // Entrainer le model
    await trainModel(model, inputs, labels);
    window.alert('Entrainement achevé!');
    setLoading(false)

    setTrainingData([model, tensorData]);
  }

  function createModel() {
    // Créer un  model sequentiel
    const model = tf.sequential();

    // Ajouter une couche d'entrées
    model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [9] }));

    // Ajouter une couche de sortie
    model.add(tf.layers.dense({ units: 1 }));

    return model;
  }

  async function trainModel(model, inputs, labels) {
    // Préparer le model à l'entrainment
    model.compile({
      optimizer: 'sgd',
      loss: 'meanSquaredError',
      metric: ['mse'],
    });


    setLoading(true);

    return await model.fit(
      inputs,
      labels,
      {
        epochs: 25,
        batchSize: 32,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance' },
            ['loss', 'mse'],
            { height: 200, callbacks: ['onEpochEnd']}
          )
      }
    )
  }



  // Récupérer les données pour le test
  function getTestData() {
    const input = tf.tensor1d([parseFloat(longit), parseFloat(latit), parseFloat(hma), parseFloat(tr), parseFloat(tb), parseFloat(pop), parseFloat(hHold), parseFloat(minc), parseFloat(prox)]);
    const minInput = input.min();
    const maxInput = input.max()
    const normalizedInput = input.sub(minInput).div(maxInput.sub(minInput));

    return {
      input: normalizedInput,
      minInput,
      maxInput,
    }
  }


  // Fonction de test
  async function testModel(model, normalizationData) {
    const { labelMin, labelMax } = normalizationData;

    const allInput = getTestData();

    const { input, minInput, maxInput } = allInput;


    // Utiliser le model pour faire une prédiction
    const inputReshaped = input.reshape([1, 9])
    const prediction = model.predict(inputReshaped);


    const unNormXs = inputReshaped
      .mul(maxInput.sub(minInput))
      .add(minInput);

    const unNormPreds = prediction
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

  


    // Comparer la prédiction a la valeur réelle
    const predictionValue = unNormPreds.dataSync();
    setValue(predictionValue);
  }
  // Test du model
  async function doTest() {
    console.log(trainingData[0]);
    testModel(trainingData[0], trainingData[1]);
  }

  function saveModel(){
    const model = trainingData[0];
    const modelJson = model.toJSON();
    const blob = new Blob([modelJson], {type: 'text/json;charset=utf-8'});
    saveAs(blob, 'model.json');
  }


  return (
    <div
      style={{
        marginLeft: 40,
        marginTop: 40,
      }}
    >
      <button style={{ borderRadius: 20, padding: 5, border: 'none', width: 200, }} onClick={() => run()}>{ !loading ? 'Entrainer le modèle' : 'Entrainement...'}</button>

      <div
        style={{
          backgroundColor: 'blue',
          width: 500,
          display: 'flex',
          height: 400,
          alignItems: 'center',
          borderRadius: 40,
          marginTop: 40,
          padding: 20,
          justifyContent: 'space-evenly',
        }}
      >
        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            width: 200,
            height: 350,
            justifyContent: 'space-around',
          }}
        >
          <input style={{ borderRadius: 20, padding: 10, border: 'none'}} type="text" placeholder="Longitude" value={longit} onChange={(text) => setLongit(text.target.value)} />
          <input style={{ borderRadius: 20, padding: 10, border: 'none'}} type="text" placeholder="Latitude" value={latit} onChange={(text) => setLatit(text.target.value)} />
          <input style={{ borderRadius: 20, padding: 10, border: 'none'}} type="text" placeholder="Home median age" value={hma} onChange={(text) => setHma(text.target.value)} />
          <input style={{ borderRadius: 20, padding: 10, border: 'none'}} type="text" placeholder="Total rooms" value={tr} onChange={(text) => setTr(text.target.value)} />
          <input style={{ borderRadius: 20, padding: 10, border: 'none'}} type="text" placeholder="Total Bedrooms" value={tb} onChange={(text) => setTb(text.target.value)} />
          <input style={{ borderRadius: 20, padding: 10, border: 'none'}} type="text" placeholder="Population" value={pop} onChange={(text) => setPop(text.target.value)} />
          <input style={{ borderRadius: 20, padding: 10, border: 'none'}} type="text" placeholder="household" value={hHold} onChange={(text) => setHHold(text.target.value)} />
          <input style={{ borderRadius: 20, padding: 10, border: 'none'}} type="text" placeholder="Median income" value={minc} onChange={(text) => setMinc(text.target.value)} />
          <input style={{ borderRadius: 20, padding: 10, border: 'none'}} type="text" placeholder="Ocean proximity" value={prox} onChange={(text) => setProx(text.target.value)} />

        </div>


        <div
          style={{
            display: 'flex',
            flexDirection: 'column',
            width: 200,
            height: 300,
            color: 'white',
          }}
        >
          <button style={{ borderRadius: 20, padding: 5, border: 'none', width: 200,}} onClick={() => doTest()}>Faire une prédiction</button>
          <p>Prédiction:  {value}</p>
        </div>

        <button style={{ position: 'absolute'  ,borderRadius: 20, padding: 5, border: 'none', width: 200, bottom: 147, left: 335,}} onClick={() => saveModel()}>Sauvegarder le modèle</button>
      </div>
    </div>
  )
}

export default App;