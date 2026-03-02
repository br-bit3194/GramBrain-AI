'use client'

import { QueryInterface } from '@/components/QueryInterface'

export default function QueryPage() {
  return (
    <div className="container-custom py-12">
      <div className="max-w-4xl mx-auto">
        <h1 className="text-4xl font-bold mb-2">Ask GramBrain AI</h1>
        <p className="text-gray-600 mb-8">
          Get intelligent recommendations for your farm. Ask about irrigation, pest management, crop planning, and more.
        </p>

        <QueryInterface />
      </div>
    </div>
  )
}
