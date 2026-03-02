'use client'

import Link from 'next/link'
import { useAppStore } from '@/store/appStore'
import { FiArrowRight, FiTrendingUp, FiActivity, FiPieChart, FiCheck } from 'react-icons/fi'

export default function Home() {
  const { user } = useAppStore()

  return (
    <div>
      {/* Hero Section with Farmland Imagery */}
      <section className="relative min-h-screen bg-gradient-to-br from-green-600 via-emerald-500 to-green-700 text-white overflow-hidden flex items-center">
        {/* Animated background pattern */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute inset-0 bg-gradient-to-b from-transparent to-green-900"></div>
          <svg className="absolute w-full h-full" viewBox="0 0 1200 600">
            <defs>
              <pattern id="wheat" x="0" y="0" width="100" height="100" patternUnits="userSpaceOnUse">
                <path d="M50,0 Q50,50 50,100" stroke="rgba(255,255,255,0.1)" strokeWidth="2" fill="none"/>
                <circle cx="50" cy="20" r="3" fill="rgba(255,255,255,0.2)"/>
                <circle cx="45" cy="35" r="2" fill="rgba(255,255,255,0.2)"/>
                <circle cx="55" cy="35" r="2" fill="rgba(255,255,255,0.2)"/>
              </pattern>
            </defs>
            <rect width="1200" height="600" fill="url(#wheat)"/>
          </svg>
        </div>

        <div className="container-custom relative z-10 py-20">
          <div className="max-w-3xl">
            <div className="mb-6 inline-block">
              <span className="bg-yellow-400 text-green-700 px-4 py-2 rounded-full text-sm font-semibold">
                🌾 AI-Powered Agriculture
              </span>
            </div>
            <h1 className="text-6xl md:text-7xl font-bold mb-6 leading-tight drop-shadow-lg">
              AI Brain for Every Farm
            </h1>
            <p className="text-xl md:text-2xl mb-8 opacity-95 drop-shadow-md max-w-2xl">
              Get intelligent, real-time recommendations for your farm using advanced AI agents and machine learning. Increase yields, reduce costs, and farm sustainably.
            </p>
            <div className="flex flex-col sm:flex-row gap-4">
              {user ? (
                <Link href="/dashboard" className="bg-yellow-400 text-green-700 px-8 py-4 rounded-lg font-bold text-lg hover:bg-yellow-300 transition shadow-lg flex items-center justify-center gap-2">
                  Go to Dashboard
                  <FiArrowRight />
                </Link>
              ) : (
                <>
                  <Link href="/register" className="bg-yellow-400 text-green-700 px-8 py-4 rounded-lg font-bold text-lg hover:bg-yellow-300 transition shadow-lg flex items-center justify-center gap-2">
                    Get Started Free
                    <FiArrowRight />
                  </Link>
                  <Link href="/login" className="border-2 border-white text-white px-8 py-4 rounded-lg font-bold text-lg hover:bg-white hover:bg-opacity-10 transition">
                    Login
                  </Link>
                </>
              )}
            </div>

            {/* Trust indicators */}
            <div className="mt-12 flex flex-col sm:flex-row gap-6 text-sm">
              <div className="flex items-center gap-2">
                <FiCheck className="text-yellow-300" size={20} />
                <span>12+ AI Agents</span>
              </div>
              <div className="flex items-center gap-2">
                <FiCheck className="text-yellow-300" size={20} />
                <span>24/7 Support</span>
              </div>
              <div className="flex items-center gap-2">
                <FiCheck className="text-yellow-300" size={20} />
                <span>100% Free</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section - 7 Services */}
      <section className="py-20 bg-white">
        <div className="container-custom">
          <div className="text-center mb-16">
            <h2 className="text-5xl font-bold mb-4">Comprehensive AI Services</h2>
            <p className="text-xl text-gray-600 max-w-2xl mx-auto">
              Seven specialized AI agents working together to optimize every aspect of your farming operations
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { icon: '🌤️', title: 'Weather Advisory', desc: '5-day forecast and real-time weather insights' },
              { icon: '🌾', title: 'Crop Advisory', desc: 'Personalized crop planning and growth guidance' },
              { icon: '🌱', title: 'Soil Analysis', desc: 'Soil health assessment and nutrient management' },
              { icon: '🐛', title: 'Pest Detection', desc: 'Early pest and disease detection with prevention' },
              { icon: '💧', title: 'Irrigation', desc: 'Optimal irrigation scheduling and water management' },
              { icon: '📈', title: 'Yield Prediction', desc: 'Accurate yield forecasting and harvest planning' },
              { icon: '💰', title: 'Market Advisory', desc: 'Real-time market prices and trading opportunities' },
              { icon: '🌍', title: 'Sustainability', desc: 'Eco-friendly practices for long-term farm health' },
            ].map((service, idx) => (
              <div key={idx} className="group card hover:shadow-xl transition-all hover:scale-105 cursor-pointer">
                <div className="text-4xl mb-4">{service.icon}</div>
                <h3 className="text-lg font-bold mb-2 group-hover:text-primary transition">{service.title}</h3>
                <p className="text-gray-600 text-sm">{service.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Why Choose Section */}
      <section className="py-20 bg-gray-50">
        <div className="container-custom">
          <h2 className="text-4xl font-bold text-center mb-16">Why Farmers Choose GramBrain AI</h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
            <div className="card hover:shadow-lg transition-shadow">
              <div className="w-14 h-14 bg-green-100 rounded-lg flex items-center justify-center mb-4">
                <FiTrendingUp className="text-green-600 text-2xl" />
              </div>
              <h3 className="text-xl font-bold mb-3">Increase Yields</h3>
              <p className="text-gray-600">
                Get data-driven recommendations to optimize crop production and maximize your harvest.
              </p>
            </div>

            <div className="card hover:shadow-lg transition-shadow">
              <div className="w-14 h-14 bg-blue-100 rounded-lg flex items-center justify-center mb-4">
                <FiActivity className="text-blue-600 text-2xl" />
              </div>
              <h3 className="text-xl font-bold mb-3">Reduce Costs</h3>
              <p className="text-gray-600">
                Optimize resource usage and reduce waste to maximize your profits and efficiency.
              </p>
            </div>

            <div className="card hover:shadow-lg transition-shadow">
              <div className="w-14 h-14 bg-emerald-100 rounded-lg flex items-center justify-center mb-4">
                <FiPieChart className="text-emerald-600 text-2xl" />
              </div>
              <h3 className="text-xl font-bold mb-3">Farm Sustainably</h3>
              <p className="text-gray-600">
                Adopt eco-friendly practices for long-term farm health and environmental responsibility.
              </p>
            </div>
          </div>

          {/* Stats */}
          <div className="bg-gradient-to-r from-green-600 to-emerald-600 text-white rounded-lg p-12">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
              <div>
                <p className="text-4xl font-bold mb-2">12+</p>
                <p className="text-green-100">AI Agents</p>
              </div>
              <div>
                <p className="text-4xl font-bold mb-2">24/7</p>
                <p className="text-green-100">Support</p>
              </div>
              <div>
                <p className="text-4xl font-bold mb-2">100%</p>
                <p className="text-green-100">Free</p>
              </div>
              <div>
                <p className="text-4xl font-bold mb-2">Real-time</p>
                <p className="text-green-100">Data</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-white">
        <div className="container-custom text-center">
          <h2 className="text-5xl font-bold mb-6">Ready to Transform Your Farm?</h2>
          <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
            Join thousands of farmers using GramBrain AI to increase yields, reduce costs, and farm sustainably. Start your free journey today.
          </p>
          {!user && (
            <Link href="/register" className="bg-green-600 text-white px-8 py-4 rounded-lg font-bold text-lg hover:bg-green-700 transition shadow-lg inline-flex items-center gap-2">
              Start Free Trial
              <FiArrowRight />
            </Link>
          )}
        </div>
      </section>
    </div>
  )
}
