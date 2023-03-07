/**
 * Recupérer le dataset et le filtrer pour obtenir les informations voulues 
 * et enlever les données incomplètes.
 */
async function getData() {
    const houseDataResponse = await fetch('http://127.0.0.1:5500/houseData.json');
    const houseData = await houseDataResponse.json();
    const cleaned = houseData.map(house => ({
        longitude: parseFloat(house.longitude),
        latitude: parseFloat(house.latitude),
        housing_median_age: parseFloat(house.housing_median_age),
        total_rooms: parseFloat(house.total_rooms),
        total_bedrooms: parseFloat(house.total_bedrooms),
        population: parseFloat(house.population),
        households: parseFloat(house.households),
        median_income: parseFloat(house.median_income),
        median_house_value: parseFloat(house.median_house_value),
    }))
        .filter(house => (house.longitude != null && house.latitude != null && house.housing_median_age != null && house.total_rooms != null  && house.median_house_value != null &&  house.households != null && house.median_income != null &&  house.population != null &&  house.total_bedrooms != null ));

    return cleaned;
}


/**
 * Convertir les données d'entrée en tensors que nous pouvons utiliser pour le machine
 * learning. Nous utiliserons aussi d'importantes méthodes comme  _shuffling_ pour brasser
 * les données et _normalizing_ pour normaliser les données
 */
function convertToTensor(data) {
    // Wrapping these calculations in a tidy will dispose any
    // intermediate tensors.

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
        ])
        const labels = data.map(d => d.median_house_value);


        const inputTensor = tf.tensor2d(inputs, [inputs.length, 8]);
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

    // Créer le model
    const model = createModel();
    tfvis.show.modelSummary({ name: 'Model Summary' }, model);

    // Recupérer les données pour l'entrainement
    const tensorData = convertToTensor(data);
    const { inputs, labels } = tensorData;

    // Entrainer le model
    await trainModel(model, inputs, labels);
    console.log('Training Done');

    // Tester le model
    testModel(model, labels, tensorData);
}

function createModel() {
    // Créer un  model sequentiel
    const model = tf.sequential();

    // Ajouter un unique champ d'entrées
    model.add(tf.layers.dense({ units: 140, activation: 'relu', inputShape: [8] }));
    
    // Ajouter un champ de sortie
    model.add(tf.layers.dense({ units: 1 }));

    return model;
}

async function trainModel(model, inputs, labels) {
    // Préparer le model à l'entrainment
    model.compile({
        optimizer: 'sgd',
        loss: 'meanSquaredError',
    });

    console.log('training');

    return await model.fit(
            inputs, 
            labels, 
            {
                epochs: 15,
                batchSize: 64,
                shuffle: true,
                callbacks: tfvis.show.fitCallbacks(
                    { name: 'Training Performance' },
                    ['loss'],
                    { height: 200, callbacks: ['onEpochEnd'] }
                )
            }
    )
}

// Récupérer les données pour le test
function getTestData() {
    const input = tf.tensor1d([-120.2, 37.26, 21.0, 2000.0, 200.1, 678.0, 127.8, 3.1819]);
    minInput = input.min();
    maxInput = input.max()
    const normalizedInput = input.sub(minInput).div(maxInput.sub(minInput));

    return {
        input: normalizedInput,
        minInput, 
        maxInput,
    }
}


// Test du model
async function testModel(model,labels, normalizationData) {
    const { labelMin, labelMax} = normalizationData;

    const allInput = getTestData();

    const { input, minInput, maxInput } = allInput;

    
    // Utiliser le model pour faire une prédiction
    const inputReshaped = input.reshape([1, 8])
    const prediction = model.predict(inputReshaped);
    console.log(prediction);

    
    const unNormXs = inputReshaped
        .mul(maxInput.sub(minInput))
        .add(minInput);
    console.log(unNormXs.dataSync());
    
    const unNormPreds = prediction
      .mul(labelMax.sub(labelMin))
      .add(labelMin);
    console.log(unNormPreds);
    
    const actualValue = labels.slice([0, 0], [1, 1])
      .mul(labelMax.sub(labelMin))
      .add(labelMin);

    
    // Comparer la prédiction a la valeur réelle
    const predictionValue = unNormPreds.dataSync();
    console.log(predictionValue);
    console.log(labels.dataSync());
    return console.log(`Predicted: ${predictionValue[0]}, Actual Value: ${actualValue.dataSync()[0]}`);
}




document.addEventListener('DOMContentLoaded', run);
