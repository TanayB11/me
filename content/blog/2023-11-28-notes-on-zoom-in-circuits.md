+++
title = "Notes: \"Zoom In - An Introduction to Circuits\""
date = 2023-11-28
description = "🌳"
+++

## Introduction
There's an uncanny overlap between the Effective Altruism and machine learning communities. Even more so in mechanistic interpretability, the subfield of ML that focuses on reverse-engineering and understanding the weights inside neural networks.

I don't consider myself to have gone that deep in to the EA realm, but I've read a few posts on [80,000 Hours](https://80000hours.org/), [LessWrong](https://www.lesswrong.com/). Some people invested in EA have started programs such as [MAIA](https://www.mitalignment.org/), [STS 10SI](https://docs.google.com/document/d/1NX0DlZRzD3NP7tBeLjMh76w7-w2s8SxV3wj0P7EYpKY/edit#heading=h.v42p15ouzzd6), and [ARENA](https://www.arena.education/). I read some of these pages and [Neel Nanda's blog](https://www.neelnanda.io/mechanistic-interpretability/quickstart), which led me to reimplement GPT-2 and explore more of the field.

One of the "foundational" papers is ["Zoom In: An Introduction to Circuits"](https://distill.pub/2020/circuits/zoom-in/) by Olah et. al. on [Distill.pub](http://distill.pub/). Below are some of my comments and notes on the paper. A lot of the content borrows directly from the source, so all figures and direct quotes are attributed to the paper and OpenAI Microscope.

## Claims 1-2: Neurons + circuits are understandable
Key ideas:
1. The possible activations of a given layer form a vector space. **Features** are directions in that vector space.
2. **Circuits** are subgraphs of the overall neural network's computational graph. We want to think of them like cells in an organism.
3. Both can be rigorously studied (quantitatively and qualitatively), and qualitative study is still important—even though most ML research focuses on maximizing quantitative benchmarks.


### Curve Detectors
> We believe that neural networks consist of meaningful, understandable features. Early layers contain features like edge or curve detectors, while later layers have features like floppy ear detectors or wheel detectors. The community is divided on whether this is true.


<center>
    <img
        src="https://openaipublic.blob.core.windows.net/microscopeprod/2020-07-25/2020-07-25/inceptionv1/lucid.feature_vis/_feature_vis/alpha%3DFalse%26negative%3DFalse%26objective%3Dneuron%26op%3Dmixed3b%253A0%26repeat%3D0%26start%3D352%26steps%3D4096%26stop%3D384/channel-379.png"
        style="border-radius: 1em;"
    ></img>
    <img
        src="https://openaipublic.blob.core.windows.net/microscopeprod/2020-07-25/2020-07-25/inceptionv1/lucid.feature_vis/_feature_vis/alpha%3DFalse%26negative%3DFalse%26objective%3Dneuron%26op%3Dmixed3b%253A0%26repeat%3D0%26start%3D384%26steps%3D4096%26stop%3D416/channel-388.png"
        style="border-radius: 1em;"
    ></img>
</center>

The authors uncovered curve detection neurons and presented several arguments as to why they detected curves—and not anything else. Note that these curve detectors have an *orientation* as well: For instance, left-facing curves did *not* highly activate the right-facing curve detector's neurons.

1. If we try to maximize activations of these neurons, we produce curves
2. The training examples that maximized the activations contained curves
3. Artificially drawn curves resulted in high activations for these neurons
4. Rotating the curves reduced the activation for one orientation's neuron, but another orientation's curve-detecting neuron started to fire
5. "We can read a curve detection algorithm off the weights"
6. "The downstream clients of curve detectors are features that naturally involve curves"
7. We can manually set weights to implement a curve detector

You can play with the activations for InceptionV1 and other models using [OpenAI's Microscope](https://microscope.openai.com/models/inceptionv1?models.technique=deep_dream).

### High-Low Frequency Detectors
However, not all features the network learns are obvious: 

> \[Some neurons\] look for low- frequency patterns on one side of their receptive field, and high-frequency patterns on the other side. Like curve detectors, high-low frequency detectors are found in families of features that look for the same thing in different orientations.
>
> They seem to be one of several heuristics for detecting the boundaries of objects, especially when the background is out of focus.

<center>
    <img
        src="https://openaipublic.blob.core.windows.net/microscopeprod/2020-07-25/2020-07-25/inceptionv1/lucid.feature_vis/_feature_vis/alpha%3DFalse%26negative%3DFalse%26objective%3Dneuron%26op%3Dmixed3a%253A0%26repeat%3D1%26start%3D96%26steps%3D4096%26stop%3D128/channel-110.png"
        style="border-radius: 1em;"
    ></img>
    <img
        src="https://openaipublic.blob.core.windows.net/microscopeprod/2020-07-25/2020-07-25/inceptionv1/lucid.feature_vis/_feature_vis/alpha%3DFalse%26negative%3DFalse%26objective%3Dneuron%26op%3Dmixed3a%253A0%26repeat%3D1%26start%3D96%26steps%3D4096%26stop%3D128/channel-112.png"
        style="border-radius: 1em;"
    ></img>
</center>

### Dog Detectors
The authors also found circuits for detecting for a left-facing dog and a right-facing dog. Those features are then combined to form orientation-invariant dog detectors. (Remember, circuits are just computational subgraphs in the neural network.)

Note: In the below figure, blue arrows represent negative weights, while red arrows represent positive weights.  The below figure is copied directly from the Distill [article](https://distill.pub/2020/circuits/zoom-in/).

<img src="/images/2023-11-28-notes-on-zoom-in-circuits/dog_detector.png" style="border-radius: 1em;">

> \[The network\] could easily create invariant neurons by not caring very much about where the eyes, fur and snout went, and just looking for a jumble of them together. But instead, the network has learned to carve apart the left and right cases and handle them separately.

The negative weights are inhibitors. Concretely: A neuron that implements a left-facing head detector looks at the fur orientation from the previous layer. If the fur is oriented left, then the activation for this left-facing head detector increases. If the fur is oriented right, then that activation is lower!

### Polysemanticity
It gets even weirder: Some circuits form a weird combination of "human-understandable features." The authors label this phenomenon the "polysemanticity of neurons." There's no reason a neural network has to "learn" in a human-like way; it's remarkable that curve detectors—and even dog detectors (as the paper goes on to discuss)—operate like this at all!

Even more oddly, neural networks can and sometimes *deliberately* introduce polysemanticity. If a neuron reads a fur detector, paw detector, and a car detector as input, it may represent a feature that is some weird combination of fur, paw, and car—because *of course* it doesn't have to behave like a human! It's a pile of linear algebra[^2]!

The authors hypothesize that this is done to effectively compress the network's representation of features. It has a finite "representation space" that is governed by its depth and the number of weights. Anthropic has an [article](https://transformer-circuits.pub/2022/toy_model/index.html) on this phenomenon, called superposition.

I came across a Twitter—I mean X—[thread](https://twitter.com/ChombaBupe/status/1727713732359229676) the other day that suggests neural networks are just compression. This [paper](https://arxiv.org/pdf/2309.10668.pdf) (that I haven't yet read) also appears to suggest the same.

This is really annoying for interpretability. Is it possible to train models without inherent polysemanticity? Perhaps we can design some loss function that looks at each neuron and discourages polysemantic neurons from forming. But to do this, we need to quantify what makes a neuron polysemantic—something that hasn't been done yet (at least according to my knowledge).

## Claim 3: Similar circuits form across models

Do neural networks in general learn in the same ways and develop similar circuits? Evidence suggests yes, but we don't have a good way of really supporting this hypothesis.

In this way, finding similar circuits across various models is analogous to finding similar traits across species to classify them. That brings us to interpretability as a natural science...

## The State of Interpretability

Maybe neural networks are too complicated, and it's unfeasible to mathematically understand *everything* that's going on at the top level. But the hope is that we can break things down into circuits—just as biologists broke down life into cells, and physicists broke up the world into atoms (and now subatomic particles).

> Circuits sidestep these challenges by focusing on tiny subgraphs of a neural network for which rigorous empirical investigation is tractable. They’re very much falsifiable: for example, if you understand a circuit, you should be able to predict what will change if you edit the weights.

Perhaps "I found this cool circuit" doesn't get published into NeurIPS or ICML, but it's still important research. If we can find *motifs*, what the authors call circuit/feature patterns across models and tasks, we will still be able to form general hypotheses and theories about how machines learn.

Because interpretability is a young niche within machine learning, there's so much low-hanging fruit. And that excites me, because it's accessible! If we want a world with human-centered, safe AI, we need interpretability. And we need to ensure that research and development are *not* centered in the hands of a powerful, elite few. That's why interpretability excites me—and if I haven't convinced you, I hope you read the entire paper to convince yourself, too.

---

[^1]: https://xkcd.com/1838/
