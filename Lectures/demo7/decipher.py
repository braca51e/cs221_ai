import util
rawText = util.toIntSeq(util.readText('lm.train'))
observations = util.toIntSeq(util.readText('ciphertext'))
print(observations)

K = 26 + 1

### Initialize the HMM

# startProbs[h]  = p_start(h)
startProbs = [1.0 / K for h in range(K)]

# transProbs[h1][h2] = p_trans(h2 | h1)
# Estimate this from raw text (fully spuervised)
transCounts = [[0 for h2 in range(K)] for h1 in range(K)]
for i in range(1, len(rawText)):
    h1, h2 = rawText[i-1], rawText[i]
    transCounts[h1][h2] += 1
transProbs = [util.normalize(counts) for counts in transCounts]
print(transProbs[util.toInt('t')][util.toInt('e')])
print(transProbs[util.toInt('t')][util.toInt('g')])

# emissionProbs[h][e] = p_emit(e | h)
emissionProbs = [[1.0 / K for e in range(K)] for h in range(K)]

### Run EM to learn the emission probabilities

for t in range(200):
    # E-step
    # q[i][h] = P(H_i = h | E = observations)
    q = util.forwardBackward(observations, startProbs, transProbs, emissionProbs)

    print(t)
    print(util.toStrSeq([util.argmax(q_i) for q_i in q]))

    # M-step
    emissionCounts = [[0 for e in range(K)] for h in range(K)]
    for i in range(len(observations)):
        for h in range(K):
            emissionCounts[h][observations[i]] += q[i][h]
    emissionProbs = [util.normalize(counts) for counts in emissionCounts]


