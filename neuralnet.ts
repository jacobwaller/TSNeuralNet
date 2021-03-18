import { OperationCanceledException } from 'typescript';
import fs from 'fs';

const randomNormal = () => {
  let r = 0;
  const v = 5;
  for (let i = v; i > 0; i--) {
    r += Math.random();
  }
  return r / v;
};

class Node {
  value: number;
  bias: number;
  inputs: Connection[];
  outputs: Connection[];
  activationFunction: (sum: number) => number;

  constructor(activationFunc: (sum: number) => number) {
    this.value = 0.0;
    this.bias = randomNormal();
    this.inputs = [];
    this.outputs = [];
    this.activationFunction = activationFunc;
  }
}

class Connection {
  toNode: Node;
  fromNode: Node;
  weight: number;

  constructor(fromNode: Node, toNode: Node) {
    this.toNode = toNode;
    this.fromNode = fromNode;
    this.weight = randomNormal();
  }
}

export default class NeuralNet {
  layers: Node[][];

  constructor(
    denseLayerNums: number[],
    activationFunctions: Array<(sum: number) => number>
  ) {
    if (denseLayerNums.length < 2) {
      console.error('Expected 2 or more layers');
      throw OperationCanceledException;
    }
    this.layers = [];
    // Create all the nodes
    denseLayerNums.forEach((layerSize, index) => {
      // Create an array of nodes of size layerSize
      const layer: Node[] = [];
      for (let i = 0; i < layerSize; i++) {
        const node = new Node(activationFunctions[index]);
        layer.push(node);
      }
      this.layers.push(layer);
    });

    // Deeply connect the nodes with weights
    for (let i = 0; i < this.layers.length - 1; i++) {
      this.layers[i].forEach((node) => {
        this.layers[i + 1].forEach((childNode) => {
          const weightBetween = new Connection(node, childNode);
          node.outputs.push(weightBetween);
          childNode.inputs.push(weightBetween);
        });
      });
    }
  }

  saveTo(filePath: string) {
    let saveString = '';
    this.layers.forEach((layer) => {
      layer.forEach((node) => {
        const bString = node.bias.toString();
        const wStrings: Array<String> = node.outputs.map((connection) =>
          connection.weight.toString()
        );

        let wString = '';
        wStrings.forEach((str) => {
          wString += `${str};`;
        });

        saveString += `${bString}&${wString}\n`;
      });
    });
    const f = fs.writeFileSync(filePath, saveString);
  }

  loadFrom(filePath: string, functions: Array<(sum: number) => number>) {
    const data = fs.readFileSync(filePath, 'utf-8');
    const lines = data.split('\n');
    let cnt = 0;
    this.layers.forEach((layer) => {
      layer.forEach((node) => {
        const line = lines[cnt];

        const bias = parseFloat(line.split('&')[0]);
        const weights = line
          .split('&')[1]
          .split(';')
          .map((item) => parseFloat(item));

        node.bias = bias;
        node.outputs.forEach((item, index) => {
          item.weight = weights[index];
        });
        cnt++;
      });
    });
  }

  predict(inputs: number[]): number[] {
    const inputLayer = this.layers[0];

    inputLayer.forEach((node, index) => {
      node.value = inputs[index];
    });

    for (let i = 1; i < this.layers.length; i++) {
      this.layers[i].forEach((node, index) => {
        let sum = 0.0;
        const inputValues = [];
        node.inputs.forEach((inputEdge) => {
          sum += inputEdge.weight * inputEdge.fromNode.value;
        });

        node.value = node.activationFunction(sum) + node.bias;
      });
    }

    return this.layers[this.layers.length - 1].map((node) => node.value);
  }
}
