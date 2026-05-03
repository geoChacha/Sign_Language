'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import Link from 'next/link';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';
import { FiArrowLeft, FiUser, FiCamera } from 'react-icons/fi';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import Select from '@/components/ui/Select';
import { useAuthStore } from '@/store/auth';
import api from '@/lib/api';

export default function ProfilePage() {
  const router = useRouter();
  const { user, isGuest, fetchUser } = useAuthStore();
  const [isLoading, setIsLoading] = useState(false);

  const [formData, setFormData] = useState({
    full_name: '',
    email: '',
    dateOfBirth: '',
    country: '',
    preferred_sign_language: 'ASL' as 'ASL' | 'PSL',
  });

  useEffect(() => {
    if (isGuest) {
      router.push('/login');
      return;
    }

    if (user) {
      setFormData({
        full_name: user.full_name || '',
        email: user.email,
        dateOfBirth: '',
        country: '',
        preferred_sign_language: user.preferred_sign_language,
      });
    }
  }, [user, isGuest, router]);

  const handleChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);

    try {
      await api.updateUser({
        full_name: formData.full_name || undefined,
        preferred_sign_language: formData.preferred_sign_language,
      });
      await fetchUser();
      toast.success('Profile updated successfully');
    } catch (error: any) {
      const message = error.response?.data?.detail || 'Failed to update profile';
      toast.error(message);
    } finally {
      setIsLoading(false);
    }
  };

  if (!user) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="loading-dots text-primary-600">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto py-8 px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <div className="flex items-center gap-4 mb-8">
            <Link
              href="/dashboard"
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <FiArrowLeft size={24} />
            </Link>
            <h1 className="text-2xl font-bold text-gray-900">Edit Profile</h1>
          </div>

          <div className="bg-white rounded-xl shadow-lg p-8">
            <div className="text-center mb-8">
              <h2 className="text-2xl font-semibold text-gray-900 mb-6">Welcome Back!</h2>

              <div className="relative inline-block">
                <div className="w-24 h-24 bg-gray-100 rounded-full flex items-center justify-center border-4 border-white shadow-lg">
                  {user.profile_picture_url ? (
                    <img
                      src={user.profile_picture_url}
                      alt={user.username}
                      className="w-full h-full rounded-full object-cover"
                    />
                  ) : (
                    <FiUser size={40} className="text-gray-400" />
                  )}
                </div>
                <button className="absolute bottom-0 right-0 p-2 bg-primary-600 text-white rounded-full hover:bg-primary-700 transition-colors">
                  <FiCamera size={16} />
                </button>
              </div>

              <h3 className="text-xl font-semibold mt-4">{user.username}</h3>
              <p className="text-gray-500 text-sm capitalize">{user.role}</p>
            </div>

            <form onSubmit={handleSubmit}>
              <h4 className="text-lg font-semibold text-gray-900 mb-4">Personal Details</h4>

              <div className="grid md:grid-cols-2 gap-6">
                <Input
                  label="Name"
                  type="text"
                  name="full_name"
                  value={formData.full_name}
                  onChange={handleChange}
                  placeholder="Your full name"
                />

                <Input
                  label="Date of Birth"
                  type="date"
                  name="dateOfBirth"
                  value={formData.dateOfBirth}
                  onChange={handleChange}
                />

                <Input
                  label="Email"
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  disabled
                  className="bg-gray-50"
                />

                <Input
                  label="Password"
                  type="password"
                  value="••••••••"
                  disabled
                  className="bg-gray-50"
                />

                <Input
                  label="Country/Region"
                  type="text"
                  name="country"
                  value={formData.country}
                  onChange={handleChange}
                  placeholder="Your country"
                />

                <Select
                  label="Preferred Sign Language"
                  name="preferred_sign_language"
                  value={formData.preferred_sign_language}
                  onChange={handleChange}
                  options={[
                    { value: 'ASL', label: 'American Sign Language (ASL)' },
                    { value: 'PSL', label: 'Pakistan Sign Language (PSL)' },
                  ]}
                />
              </div>

              <div className="mt-8 flex justify-center">
                <Button type="submit" isLoading={isLoading} className="px-12">
                  Save Changes
                </Button>
              </div>
            </form>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
