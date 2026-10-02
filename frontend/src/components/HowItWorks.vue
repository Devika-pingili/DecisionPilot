<script setup>
import { ArrowRight, Boxes, BrainCircuit, Clock3, PackageSearch, ShoppingBag } from '@lucide/vue'

const steps = [
  { label: 'Purchase history', icon: ShoppingBag, tint: 'mint' },
  { label: 'Point-in-time features', icon: Clock3, tint: 'lime' },
  { label: 'Candidate generation', icon: PackageSearch, tint: 'peach' },
  { label: 'ML ranking', icon: BrainCircuit, tint: 'blue' },
  { label: 'Personalized picks', icon: Boxes, tint: 'green' },
]
</script>

<template>
  <section class="process-section" aria-labelledby="process-heading">
    <div class="section-heading process-heading">
      <div>
        <p class="eyebrow">Behind the ranking</p>
        <h2 id="process-heading">How it works</h2>
      </div>
      <p class="process-caption">One point-in-time pipeline, from history to shortlist.</p>
    </div>

    <ol class="process-flow">
      <template v-for="(step, index) in steps" :key="step.label">
        <li class="process-step">
          <span class="process-icon" :class="`tint-${step.tint}`">
            <component :is="step.icon" :size="18" :stroke-width="1.8" />
          </span>
          <span>{{ step.label }}</span>
        </li>
        <li v-if="index < steps.length - 1" class="process-arrow" aria-hidden="true">
          <ArrowRight :size="15" :stroke-width="1.8" />
        </li>
      </template>
    </ol>
  </section>
</template>