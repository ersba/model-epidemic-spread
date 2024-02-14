
°&root"_tf_keras_layer*Å&{
  "name": "sequential",
  "trainable": true,
  "expects_training_arg": false,
  "dtype": "float32",
  "batch_input_shape": null,
  "autocast": false,
  "class_name": "Sequential",
  "config": {
    "name": "sequential",
    "layers": [
      {
        "name": "dense_input",
        "class_name": "InputLayer",
        "config": {
          "sparse": false,
          "ragged": false,
          "name": "dense_input",
          "dtype": "float32",
          "batch_input_shape": {
            "class_name": "TensorShape",
            "items": [
              null,
              1
            ]
          }
        },
        "inbound_nodes": []
      },
      {
        "name": "dense",
        "class_name": "Dense",
        "config": {
          "units": 32,
          "activation": "linear",
          "use_bias": true,
          "kernel_initializer": {
            "class_name": "GlorotUniform",
            "config": {
              "seed": null
            }
          },
          "bias_initializer": {
            "class_name": "Zeros",
            "config": {}
          },
          "kernel_regularizer": null,
          "bias_regularizer": null,
          "kernel_constraint": null,
          "bias_constraint": null,
          "name": null,
          "dtype": "float32",
          "trainable": true
        },
        "inbound_nodes": [
          [
            "dense_input",
            0,
            0
          ]
        ]
      },
      {
        "name": "leaky_re_lu",
        "class_name": "LeakyReLu",
        "config": {
          "alpha": 0.01,
          "name": null,
          "dtype": "float32",
          "trainable": true
        },
        "inbound_nodes": [
          [
            "dense",
            0,
            0
          ]
        ]
      },
      {
        "name": "dense_1",
        "class_name": "Dense",
        "config": {
          "units": 64,
          "activation": "linear",
          "use_bias": true,
          "kernel_initializer": {
            "class_name": "GlorotUniform",
            "config": {
              "seed": null
            }
          },
          "bias_initializer": {
            "class_name": "Zeros",
            "config": {}
          },
          "kernel_regularizer": null,
          "bias_regularizer": null,
          "kernel_constraint": null,
          "bias_constraint": null,
          "name": null,
          "dtype": "float32",
          "trainable": true
        },
        "inbound_nodes": [
          [
            "leaky_re_lu",
            0,
            0
          ]
        ]
      },
      {
        "name": "leaky_re_lu_1",
        "class_name": "LeakyReLu",
        "config": {
          "alpha": 0.01,
          "name": null,
          "dtype": "float32",
          "trainable": true
        },
        "inbound_nodes": [
          [
            "dense_1",
            0,
            0
          ]
        ]
      },
      {
        "name": "dense_2",
        "class_name": "Dense",
        "config": {
          "units": 5,
          "activation": "sigmoid",
          "use_bias": true,
          "kernel_initializer": {
            "class_name": "GlorotUniform",
            "config": {
              "seed": null
            }
          },
          "bias_initializer": {
            "class_name": "Zeros",
            "config": {}
          },
          "kernel_regularizer": null,
          "bias_regularizer": null,
          "kernel_constraint": null,
          "bias_constraint": null,
          "name": null,
          "dtype": "float32",
          "trainable": true
        },
        "inbound_nodes": [
          [
            "leaky_re_lu_1",
            0,
            0
          ]
        ]
      }
    ],
    "input_layers": [
      [
        "dense_input",
        0,
        0
      ],
      [
        "dense_input",
        0,
        0
      ],
      [
        "dense_input",
        0,
        0
      ],
      [
        "dense_input",
        0,
        0
      ],
      [
        "dense_input",
        0,
        0
      ],
      [
        "dense_input",
        0,
        0
      ]
    ],
    "output_layers": [
      [
        "dense_input",
        0,
        0
      ],
      [
        "dense",
        0,
        0
      ],
      [
        "leaky_re_lu",
        0,
        0
      ],
      [
        "dense_1",
        0,
        0
      ],
      [
        "leaky_re_lu_1",
        0,
        0
      ],
      [
        "dense_2",
        0,
        0
      ]
    ]
  },
  "shared_object_id": 1,
  "build_input_shape": {
    "class_name": "TensorShape",
    "items": [
      null,
      1
    ]
  }
}2
Ëroot.layer_with_weights-0"_tf_keras_layer*±{
  "name": "dense",
  "trainable": true,
  "expects_training_arg": false,
  "dtype": "float32",
  "batch_input_shape": null,
  "autocast": false,
  "input_spec": {
    "class_name": "InputSpec",
    "config": {
      "DType": null,
      "Shape": null,
      "Ndim": null,
      "MinNdim": 2,
      "MaxNdim": null,
      "Axes": {
        "-1": 1
      }
    },
    "shared_object_id": 2
  },
  "class_name": "Dense",
  "config": {
    "units": 32,
    "activation": "linear",
    "use_bias": true,
    "kernel_initializer": {
      "class_name": "GlorotUniform",
      "config": {
        "seed": null
      }
    },
    "bias_initializer": {
      "class_name": "Zeros",
      "config": {}
    },
    "kernel_regularizer": null,
    "bias_regularizer": null,
    "kernel_constraint": null,
    "bias_constraint": null,
    "name": null,
    "dtype": "float32",
    "trainable": true
  },
  "shared_object_id": 3,
  "build_input_shape": {
    "class_name": "TensorShape",
    "items": [
      null,
      1
    ]
  }
}2
ﬁroot.layer-1"_tf_keras_layer*¥{
  "name": "leaky_re_lu",
  "trainable": true,
  "expects_training_arg": false,
  "dtype": "float32",
  "batch_input_shape": null,
  "autocast": false,
  "class_name": "LeakyReLu",
  "config": {
    "alpha": 0.01,
    "name": null,
    "dtype": "float32",
    "trainable": true
  },
  "shared_object_id": 4,
  "build_input_shape": {
    "class_name": "TensorShape",
    "items": [
      null,
      32
    ]
  }
}2
Ïroot.layer_with_weights-1"_tf_keras_layer*µ{
  "name": "dense_1",
  "trainable": true,
  "expects_training_arg": false,
  "dtype": "float32",
  "batch_input_shape": null,
  "autocast": false,
  "input_spec": {
    "class_name": "InputSpec",
    "config": {
      "DType": null,
      "Shape": null,
      "Ndim": null,
      "MinNdim": 2,
      "MaxNdim": null,
      "Axes": {
        "-1": 32
      }
    },
    "shared_object_id": 5
  },
  "class_name": "Dense",
  "config": {
    "units": 64,
    "activation": "linear",
    "use_bias": true,
    "kernel_initializer": {
      "class_name": "GlorotUniform",
      "config": {
        "seed": null
      }
    },
    "bias_initializer": {
      "class_name": "Zeros",
      "config": {}
    },
    "kernel_regularizer": null,
    "bias_regularizer": null,
    "kernel_constraint": null,
    "bias_constraint": null,
    "name": null,
    "dtype": "float32",
    "trainable": true
  },
  "shared_object_id": 6,
  "build_input_shape": {
    "class_name": "TensorShape",
    "items": [
      null,
      32
    ]
  }
}2
‡root.layer-3"_tf_keras_layer*∂{
  "name": "leaky_re_lu_1",
  "trainable": true,
  "expects_training_arg": false,
  "dtype": "float32",
  "batch_input_shape": null,
  "autocast": false,
  "class_name": "LeakyReLu",
  "config": {
    "alpha": 0.01,
    "name": null,
    "dtype": "float32",
    "trainable": true
  },
  "shared_object_id": 7,
  "build_input_shape": {
    "class_name": "TensorShape",
    "items": [
      null,
      64
    ]
  }
}2
Ïroot.layer_with_weights-2"_tf_keras_layer*µ{
  "name": "dense_2",
  "trainable": true,
  "expects_training_arg": false,
  "dtype": "float32",
  "batch_input_shape": null,
  "autocast": false,
  "input_spec": {
    "class_name": "InputSpec",
    "config": {
      "DType": null,
      "Shape": null,
      "Ndim": null,
      "MinNdim": 2,
      "MaxNdim": null,
      "Axes": {
        "-1": 64
      }
    },
    "shared_object_id": 8
  },
  "class_name": "Dense",
  "config": {
    "units": 5,
    "activation": "sigmoid",
    "use_bias": true,
    "kernel_initializer": {
      "class_name": "GlorotUniform",
      "config": {
        "seed": null
      }
    },
    "bias_initializer": {
      "class_name": "Zeros",
      "config": {}
    },
    "kernel_regularizer": null,
    "bias_regularizer": null,
    "kernel_constraint": null,
    "bias_constraint": null,
    "name": null,
    "dtype": "float32",
    "trainable": true
  },
  "shared_object_id": 9,
  "build_input_shape": {
    "class_name": "TensorShape",
    "items": [
      null,
      64
    ]
  }
}2