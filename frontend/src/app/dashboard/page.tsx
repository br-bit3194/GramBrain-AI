'use client'

import { useAppStore } from '@/store/appStore'
import Link from 'next/link'
import { FiArrowRight, FiCloud, FiActivity, FiDroplet, FiBarChart2, FiTrendingUp, FiDollarSign } from 'react-icons/fi'
import { useState, useEffect } from 'react'

export default function DashboardPage() {
  const { user, farm } = useAppStore()
  const [scrollY, setScrollY] = useState(0)

  useEffect(() => {
    const handleScroll = () => setScrollY(window.scrollY)
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  if (!user) {
    return (
      <div className="container-custom py-12">
        <div className="card text-center">
          <h2 className="text-2xl font-bold mb-4">Please Login</h2>
          <p className="text-gray-600 mb-6">You need to login to access the dashboard.</p>
          <Link href="/login" className="btn-primary inline-block">
            Go to Login
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div>
      {/* Hero Section with Farmland Imagery */}
      <section 
        className="relative h-96 bg-gradient-to-r from-green-600 via-green-500 to-emerald-600 text-white overflow-hidden"
        style={{
          backgroundImage: 'url("data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 1200 400%22%3E%3Cpath d=%22M0,200 Q300,100 600,200 T1200,200 L1200,400 L0,400 Z%22 fill=%22rgba(255,255,255,0.1)%22/%3E%3Cpath d=%22M0,250 Q300,150 600,250 T1200,250 L1200,400 L0,400 Z%22 fill=%22rgba(255,255,255,0.05)%22/%3E%3C/svg%3E")',
          backgroundSize: 'cover',
          backgroundPosition: 'center',
        }}
      >
        {/* Parallax effect */}
        <div 
          className="absolute inset-0 opacity-20"
          style={{
            transform: `translateY(${scrollY * 0.5}px)`,
          }}
        >
          <div className="absolute inset-0 bg-gradient-to-b from-transparent to-green-900"></div>
        </div>

        <div className="container-custom relative z-10 h-full flex flex-col justify-center">
          <h1 className="text-5xl md:text-6xl font-bold mb-4 drop-shadow-lg">
            Welcome, {user.name}! 🌾
          </h1>
          <p className="text-xl md:text-2xl opacity-90 drop-shadow-md max-w-2xl">
            Your AI-powered farm intelligence dashboard
          </p>
          {farm && (
            <p className="text-lg opacity-80 mt-4 drop-shadow-md">
              📍 {farm.area_hectares} hectares • {farm.soil_type} soil • {farm.irrigation_type} irrigation
            </p>
          )}
        </div>
      </section>

      {/* Quick Stats */}
      <section className="container-custom py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <div className="card hover:shadow-lg transition-shadow">
            <h3 className="text-gray-600 text-sm font-medium mb-2">Active Farm</h3>
            <p className="text-3xl font-bold text-primary">{farm?.area_hectares || 0} ha</p>
            <p className="text-xs text-gray-500 mt-2">Total farm area</p>
          </div>
          <div className="card hover:shadow-lg transition-shadow">
            <h3 className="text-gray-600 text-sm font-medium mb-2">Soil Type</h3>
            <p className="text-3xl font-bold text-primary capitalize">{farm?.soil_type || 'N/A'}</p>
            <p className="text-xs text-gray-500 mt-2">Current soil composition</p>
          </div>
          <div className="card hover:shadow-lg transition-shadow">
            <h3 className="text-gray-600 text-sm font-medium mb-2">Irrigation</h3>
            <p className="text-3xl font-bold text-primary capitalize">{farm?.irrigation_type || 'N/A'}</p>
            <p className="text-xs text-gray-500 mt-2">Water management system</p>
          </div>
        </div>
      </section>

      {/* Service Cards - 7 Services */}
      <section className="bg-gray-50 py-16">
        <div className="container-custom">
          <h2 className="text-4xl font-bold mb-4 text-center">AI-Powered Services</h2>
          <p className="text-center text-gray-600 mb-12 max-w-2xl mx-auto">
            Get intelligent recommendations across all aspects of your farming operations
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {/* Weather Service */}
            <Link href="/query" className="group">
              <div className="card h-full hover:shadow-xl transition-all hover:scale-105 cursor-pointer">
                <div className="w-14 h-14 bg-blue-100 rounded-lg flex items-center justify-center mb-4 group-hover:bg-blue-200 transition">
                  <FiCloud className="text-blue-600 text-2xl" />
                </div>
                <h3 className="text-lg font-bold mb-2">Weather Advisory</h3>
                <p className="text-gray-600 text-sm mb-4">
                  5-day forecast and real-time weather insights for your farm
                </p>
                <span className="text-primary text-sm font-semibold flex items-center gap-1">
                  Learn more <FiArrowRight size={16} />
                </span>
              </div>
            </Link>

            {/* Crop Advisory */}
            <Link href="/query" className="group">
              <div className="card h-full hover:shadow-xl transition-all hover:scale-105 cursor-pointer">
                <div className="w-14 h-14 bg-green-100 rounded-lg flex items-center justify-center mb-4 group-hover:bg-green-200 transition">
                  <FiActivity className="text-green-600 text-2xl" />
                </div>
                <h3 className="text-lg font-bold mb-2">Crop Advisory</h3>
                <p className="text-gray-600 text-sm mb-4">
                  Personalized crop planning and growth stage recommendations
                </p>
                <span className="text-primary text-sm font-semibold flex items-center gap-1">
                  Learn more <FiArrowRight size={16} />
                </span>
              </div>
            </Link>

            {/* Soil Analysis */}
            <Link href="/query" className="group">
              <div className="card h-full hover:shadow-xl transition-all hover:scale-105 cursor-pointer">
                <div className="w-14 h-14 bg-amber-100 rounded-lg flex items-center justify-center mb-4 group-hover:bg-amber-200 transition">
                  <FiBarChart2 className="text-amber-600 text-2xl" />
                </div>
                <h3 className="text-lg font-bold mb-2">Soil Analysis</h3>
                <p className="text-gray-600 text-sm mb-4">
                  Soil health assessment and nutrient management guidance
                </p>
                <span className="text-primary text-sm font-semibold flex items-center gap-1">
                  Learn more <FiArrowRight size={16} />
                </span>
              </div>
            </Link>

            {/* Pest & Disease Detection */}
            <Link href="/query" className="group">
              <div className="card h-full hover:shadow-xl transition-all hover:scale-105 cursor-pointer">
                <div className="w-14 h-14 bg-red-100 rounded-lg flex items-center justify-center mb-4 group-hover:bg-red-200 transition">
                  <FiBarChart2 className="text-red-600 text-2xl" />
                </div>
                <h3 className="text-lg font-bold mb-2">Pest Detection</h3>
                <p className="text-gray-600 text-sm mb-4">
                  Early pest and disease detection with prevention strategies
                </p>
                <span className="text-primary text-sm font-semibold flex items-center gap-1">
                  Learn more <FiArrowRight size={16} />
                </span>
              </div>
            </Link>

            {/* Irrigation Management */}
            <Link href="/query" className="group">
              <div className="card h-full hover:shadow-xl transition-all hover:scale-105 cursor-pointer">
                <div className="w-14 h-14 bg-cyan-100 rounded-lg flex items-center justify-center mb-4 group-hover:bg-cyan-200 transition">
                  <FiDroplet className="text-cyan-600 text-2xl" />
                </div>
                <h3 className="text-lg font-bold mb-2">Irrigation</h3>
                <p className="text-gray-600 text-sm mb-4">
                  Optimal irrigation scheduling and water management
                </p>
                <span className="text-primary text-sm font-semibold flex items-center gap-1">
                  Learn more <FiArrowRight size={16} />
                </span>
              </div>
            </Link>

            {/* Yield Prediction */}
            <Link href="/query" className="group">
              <div className="card h-full hover:shadow-xl transition-all hover:scale-105 cursor-pointer">
                <div className="w-14 h-14 bg-purple-100 rounded-lg flex items-center justify-center mb-4 group-hover:bg-purple-200 transition">
                  <FiTrendingUp className="text-purple-600 text-2xl" />
                </div>
                <h3 className="text-lg font-bold mb-2">Yield Prediction</h3>
                <p className="text-gray-600 text-sm mb-4">
                  Accurate yield forecasting and harvest planning
                </p>
                <span className="text-primary text-sm font-semibold flex items-center gap-1">
                  Learn more <FiArrowRight size={16} />
                </span>
              </div>
            </Link>

            {/* Market Advisory */}
            <Link href="/marketplace" className="group">
              <div className="card h-full hover:shadow-xl transition-all hover:scale-105 cursor-pointer">
                <div className="w-14 h-14 bg-orange-100 rounded-lg flex items-center justify-center mb-4 group-hover:bg-orange-200 transition">
                  <FiDollarSign className="text-orange-600 text-2xl" />
                </div>
                <h3 className="text-lg font-bold mb-2">Market Advisory</h3>
                <p className="text-gray-600 text-sm mb-4">
                  Real-time market prices and trading opportunities
                </p>
                <span className="text-primary text-sm font-semibold flex items-center gap-1">
                  Learn more <FiArrowRight size={16} />
                </span>
              </div>
            </Link>
          </div>
        </div>
      </section>

      {/* Weather Section with 5-day Forecast */}
      <section className="container-custom py-16">
        <h2 className="text-4xl font-bold mb-4">5-Day Weather Forecast</h2>
        <p className="text-gray-600 mb-8">Live weather data for your farm location</p>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
          {[
            { day: 'Today', temp: '28°C', condition: '☀️ Sunny', humidity: '65%' },
            { day: 'Tomorrow', temp: '26°C', condition: '⛅ Partly Cloudy', humidity: '70%' },
            { day: 'Day 3', temp: '24°C', condition: '🌧️ Rainy', humidity: '85%' },
            { day: 'Day 4', temp: '25°C', condition: '⛅ Partly Cloudy', humidity: '75%' },
            { day: 'Day 5', temp: '27°C', condition: '☀️ Sunny', humidity: '60%' },
          ].map((forecast, idx) => (
            <div key={idx} className="card text-center hover:shadow-lg transition-shadow">
              <p className="font-bold text-lg mb-2">{forecast.day}</p>
              <p className="text-3xl mb-2">{forecast.temp}</p>
              <p className="text-sm text-gray-600 mb-2">{forecast.condition}</p>
              <p className="text-xs text-gray-500">Humidity: {forecast.humidity}</p>
            </div>
          ))}
        </div>
      </section>

      {/* About Section with Key Stats */}
      <section className="bg-gradient-to-r from-primary to-green-600 text-white py-16">
        <div className="container-custom">
          <h2 className="text-4xl font-bold mb-12 text-center">Why Farmers Love GramBrain AI</h2>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-12">
            <div className="text-center">
              <p className="text-4xl font-bold mb-2">12+</p>
              <p className="text-lg opacity-90">AI Agents</p>
              <p className="text-sm opacity-75">Specialized for different farming needs</p>
            </div>
            <div className="text-center">
              <p className="text-4xl font-bold mb-2">24/7</p>
              <p className="text-lg opacity-90">Support</p>
              <p className="text-sm opacity-75">Always available for your questions</p>
            </div>
            <div className="text-center">
              <p className="text-4xl font-bold mb-2">100%</p>
              <p className="text-lg opacity-90">Free</p>
              <p className="text-sm opacity-75">No hidden charges or subscriptions</p>
            </div>
            <div className="text-center">
              <p className="text-4xl font-bold mb-2">Real-time</p>
              <p className="text-lg opacity-90">Data</p>
              <p className="text-sm opacity-75">Live weather and market insights</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div className="bg-white bg-opacity-10 p-6 rounded-lg backdrop-blur">
              <h3 className="text-xl font-bold mb-3">🎯 Precision Farming</h3>
              <p className="opacity-90">
                Get exact recommendations tailored to your farm's unique conditions and needs
              </p>
            </div>
            <div className="bg-white bg-opacity-10 p-6 rounded-lg backdrop-blur">
              <h3 className="text-xl font-bold mb-3">💰 Cost Reduction</h3>
              <p className="opacity-90">
                Optimize resource usage and reduce waste to maximize your profits
              </p>
            </div>
            <div className="bg-white bg-opacity-10 p-6 rounded-lg backdrop-blur">
              <h3 className="text-xl font-bold mb-3">🌱 Sustainability</h3>
              <p className="opacity-90">
                Adopt eco-friendly practices for long-term farm health and productivity
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Quick Actions */}
      <section className="container-custom py-16">
        <h2 className="text-4xl font-bold mb-12 text-center">Quick Actions</h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="card hover:shadow-lg transition-shadow">
            <h3 className="text-2xl font-bold mb-4">💬 Ask AI Questions</h3>
            <p className="text-gray-600 mb-6">
              Get instant recommendations for any farming challenge you're facing.
            </p>
            <Link href="/query" className="btn-primary inline-flex items-center gap-2">
              Ask Now
              <FiArrowRight />
            </Link>
          </div>

          <div className="card hover:shadow-lg transition-shadow">
            <h3 className="text-2xl font-bold mb-4">🌾 Manage Farms</h3>
            <p className="text-gray-600 mb-6">
              Add new farms or update your existing farm information and settings.
            </p>
            <Link href="/farms" className="btn-primary inline-flex items-center gap-2">
              Manage Farms
              <FiArrowRight />
            </Link>
          </div>

          <div className="card hover:shadow-lg transition-shadow">
            <h3 className="text-2xl font-bold mb-4">🛒 Marketplace</h3>
            <p className="text-gray-600 mb-6">
              Browse products, list your harvest, and connect with buyers and sellers.
            </p>
            <Link href="/marketplace" className="btn-primary inline-flex items-center gap-2">
              Browse Products
              <FiArrowRight />
            </Link>
          </div>

          <div className="card hover:shadow-lg transition-shadow">
            <h3 className="text-2xl font-bold mb-4">👤 Profile</h3>
            <p className="text-gray-600 mb-6">
              Update your profile, preferences, and account settings.
            </p>
            <Link href="/profile" className="btn-primary inline-flex items-center gap-2">
              Edit Profile
              <FiArrowRight />
            </Link>
          </div>
        </div>
      </section>
    </div>
  )
}
