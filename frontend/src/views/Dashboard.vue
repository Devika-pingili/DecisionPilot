<script setup>
import { computed, onMounted, ref } from 'vue'
import { ArrowRight, Clock3, UserRound } from '@lucide/vue'
import { explainRecommendation as requestRecommendationExplanation, getCustomerSummary, getHealth, getRecommendations } from '../services/api'
import CustomerSummary from '../components/CustomerSummary.vue'
import CustomerInsights from '../components/CustomerInsights.vue'
import Header from '../components/Header.vue'
import HowItWorks from '../components/HowItWorks.vue'
import RecommendationList from '../components/RecommendationList.vue'

const customerId = ref(1)
const topK = ref(5)
const orderNumber = ref('')
const summary = ref(null)
const recommendations = ref([])
const isLoading = ref(false)
const summaryLoading = ref(true)
const summaryError = ref('')
const recommendationsError = ref('')
const aiExplanation = ref(null)
const aiExplanationLoading = ref(false)
const backendHealthy = ref(false)
const requestSequence = ref(0)
const explanationRequestSequence = ref(0)

const resultCountLabel = computed(() => {
  const count = recommendations.value.length
  return `${String(count).padStart(2, '0')} ${count === 1 ? 'product' : 'products'}`
})

async function generateRecommendations() {
  const normalizedCustomerId = Number(customerId.value)
  if (!Number.isSafeInteger(normalizedCustomerId) || normalizedCustomerId < 1) {
    recommendationsError.value = 'Enter a valid customer ID greater than zero.'
    recommendations.value = []
    summary.value = null
    summaryError.value = ''
    return
  }

  const normalizedOrderNumber = orderNumber.value === '' ? null : Number(orderNumber.value)
  if (normalizedOrderNumber !== null && (!Number.isSafeInteger(normalizedOrderNumber) || normalizedOrderNumber < 1)) {
    recommendationsError.value = 'Enter a valid order number greater than zero, or leave it blank.'
    recommendations.value = []
    return
  }

  const requestId = ++requestSequence.value
  explanationRequestSequence.value += 1
  aiExplanation.value = null
  aiExplanationLoading.value = false
  isLoading.value = true
  summaryLoading.value = true
  summaryError.value = ''
  recommendationsError.value = ''
  backendHealthy.value = false

  const [healthResult, summaryResult, recommendationsResult] = await Promise.allSettled([
    getHealth(),
    getCustomerSummary(normalizedCustomerId),
    getRecommendations(normalizedCustomerId, topK.value, normalizedOrderNumber),
  ])

  if (requestId !== requestSequence.value) return

  backendHealthy.value = healthResult.status === 'fulfilled' && healthResult.value.status === 'healthy'
  summaryLoading.value = false
  if (summaryResult.status === 'fulfilled') {
    summary.value = summaryResult.value
  } else {
    summary.value = null
    summaryError.value = summaryResult.reason?.message || 'Unable to load customer history.'
  }

  if (recommendationsResult.status === 'fulfilled') {
    recommendations.value = recommendationsResult.value.recommendations
  } else {
    recommendations.value = []
    recommendationsError.value = recommendationsResult.reason?.message
      || 'Unable to load recommendations. Please make sure the DecisionPilot backend is running.'
  }
  isLoading.value = false
}

async function explainRecommendation(recommendation) {
  if (!recommendation || isLoading.value || aiExplanationLoading.value) return

  const requestId = ++explanationRequestSequence.value
  aiExplanation.value = null
  aiExplanationLoading.value = true

  try {
    aiExplanation.value = await requestRecommendationExplanation(
      Number(customerId.value),
      recommendation.product_id,
      topK.value,
      orderNumber.value === '' ? null : Number(orderNumber.value),
    )
  } catch {
    if (requestId === explanationRequestSequence.value) {
      aiExplanation.value = {
        available: false,
        deterministic_explanation: recommendation.explanation_short,
        message: 'AI explanation is currently unavailable.',
      }
    }
  } finally {
    if (requestId === explanationRequestSequence.value) {
      aiExplanationLoading.value = false
    }
  }
}

onMounted(() => {
  generateRecommendations()
})
</script>

<template>
  <div id="top" class="dashboard-shell">
    <Header :backend-healthy="backendHealthy" :customer-id="summary?.customer_id ?? customerId" />

    <main class="dashboard-main">
      <section class="hero-panel" aria-labelledby="hero-title">
        <div class="hero-copy">
          <p class="hero-kicker"><span></span> NEXT BASKET INTELLIGENCE</p>
          <h1 id="hero-title">Predict your<br /><em>next basket.</em></h1>
          <p class="hero-description">
            Discover products you are most likely to need, guided by your purchase history.
          </p>
          <div class="hero-footnote">
            <Clock3 :size="15" :stroke-width="1.8" />
            <span>Every recommendation is grounded in history before the selected order.</span>
          </div>
        </div>

        <form class="prediction-form" @submit.prevent="generateRecommendations">
          <div class="form-heading">
            <div>
              <p class="form-overline">PREDICTION SETUP</p>
              <h2>Build a shortlist</h2>
            </div>
            <span class="form-index">01 / 03</span>
          </div>

          <label class="field-label" for="customer-id">Customer ID</label>
          <div class="input-shell customer-input-shell">
            <UserRound :size="17" :stroke-width="1.8" aria-hidden="true" />
            <input
              id="customer-id"
              v-model.number="customerId"
              type="number"
              inputmode="numeric"
              min="1"
              step="1"
              autocomplete="off"
              aria-describedby="customer-help"
            />
            <span class="input-context">DEMO CUSTOMER</span>
          </div>
          <p id="customer-help" class="field-help">Use a customer ID from the order history.</p>

          <div class="form-row">
            <div class="form-field">
              <span class="field-label" id="top-k-label">Top K</span>
              <div class="segmented-control" role="group" aria-labelledby="top-k-label">
                <button
                  v-for="count in [5, 10, 20]"
                  :key="count"
                  type="button"
                  :aria-pressed="topK === count"
                  :class="{ selected: topK === count }"
                  @click="topK = count"
                >
                  {{ count }}
                </button>
              </div>
            </div>
            <div class="form-field cutoff-field">
              <label class="field-label" for="order-number">Order cutoff <span>Optional</span></label>
              <input
                id="order-number"
                v-model="orderNumber"
                class="cutoff-input"
                type="number"
                inputmode="numeric"
                min="1"
                step="1"
                placeholder="Latest"
              />
            </div>
          </div>

          <button class="generate-button" type="submit" :disabled="isLoading">
            <span>{{ isLoading ? 'Generating shortlist...' : 'Generate recommendations' }}</span>
            <ArrowRight v-if="!isLoading" :size="17" :stroke-width="2" />
            <span v-else class="button-spinner" aria-hidden="true"></span>
          </button>
          <p class="form-assurance">Ranked by DecisionPilot’s existing recommendation model</p>
        </form>
      </section>

      <CustomerSummary
        :summary="summary"
        :loading="summaryLoading"
        :error="summaryError"
      />

      <section class="recommendations-section" aria-labelledby="recommendations-heading">
        <div class="section-heading recommendations-heading">
          <div>
            <p class="eyebrow">Personalized shortlist</p>
            <h2 id="recommendations-heading">Recommended for you</h2>
            <p class="section-description">A ranked view of products the model scores highest for this next order.</p>
          </div>
          <span v-if="!isLoading && !recommendationsError" class="result-count">{{ resultCountLabel }}</span>
        </div>

        <RecommendationList
          :recommendations="recommendations"
          :loading="isLoading"
          :error="recommendationsError"
          @retry="generateRecommendations"
        />
        <p v-if="recommendations.length && !isLoading" class="score-note">
          Recommendation scores support ranking; they are not calibrated probabilities.
        </p>
      </section>

      <CustomerInsights
        :summary="summary"
        :recommendations="recommendations"
        :loading="isLoading"
        :error="recommendationsError"
        :ai-explanation="aiExplanation"
        :ai-loading="aiExplanationLoading"
        @explain="explainRecommendation"
      />

      <HowItWorks />
    </main>

    <footer class="app-footer">
      <span>DECISIONPILOT</span>
      <p>Point-in-time history in. Better next baskets out.</p>
    </footer>
  </div>
</template>